import sys
import asyncio
import enum
from collections import deque
import math

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


class State(enum.Enum):
    # enum.auto() 自动给 1/2/3(从 1 递增)
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


class Request:

    def __init__(self, prompt_ids, max_tokens, eos_token_id):
        # list[int]
        self.prompt_ids = prompt_ids
        # list[int](自回归生成的 token id,逐个 append)
        self.generated_ids = []
        # xstar_cpp.PagedKVCache(每请求一个)
        self.kv = None
        # 枚举 WAITING / RUNNING / FINISHED
        self.state = State.WAITING
        # int
        self.max_tokens = max_tokens
        # int
        self.eos_token_id = eos_token_id
        # asyncio.Queue  调度器 put_nowait 每步 token, generator await get 拿, 流式解耦
        self.token_queue = asyncio.Queue()
        # match 返回的节点
        self.radix_tree_node = None
        # self.radix_matched = 0  # 命中块数,仅日志用


class NoRadixTree:
    # 空对象桩
    # 方法面与 RadixTree pybind 绑定一一对齐，签名改动必须两边同步

    def match_prefix(self, tokens):
        # 0 token 命中 → needed_new = 全量 → 退化成老路径
        return ([], None)

    def insert(self, tokens, block_table, bm):
        # 树里什么都没记 → finish 不持有任何块
        return None

    def evict(self, need_blocks, bm):
        # 一块都驱不出来 → 准入判据把 evictable 当 0 → 必须靠 reset
        return 0

    def inc_lock_ref(self, node):
        # 没有 pin 可加
        pass

    def dec_lock_ref(self, node):
        # 没有 pin 可减
        pass

    def lru_size(self):
        # 测试/日志用，桩永远空
        return 0

    def evictable_blocks(self):
        # 没有可驱逐的块
        return 0


class Scheduler:
    def __init__(self, worker, use_radix=False, decode_ratio=1.0, decode_cap=512):
        self.worker = worker
        self.waiting = deque()
        self.running = deque()
        self._wake = asyncio.Event()  # 唤醒空转的循环
        self._task = None  # 后台自己跑的协程
        self.radix_tree = (
            xstar_cpp.RadixTree(worker.block_size) if use_radix else NoRadixTree()
        )
        self.decode_ratio = decode_ratio  # decode 预留比例
        self.decode_cap = decode_cap  # 单个请求的预留最多 token 数
        self._aborted = set()
        # 连续失败次数
        # 两种失败，处置不同
        # 偶发失败（某一批请求碰上 OOM）：救完人继续跑是对的——服务还有别的请求要处理，为一批坏请求把整个调度器弄死，等于全服务陪葬
        # 持续性失败（权重坏了、GPU 掉了）：每轮 tick 必崩，救人→继续→再崩无限循环——这种硬撑没意义，应该让它死得响亮（re-raise 出去，让进程层面看到）
        self.fail_streak = 0

    def abort(self, req):
        # 把 id 塞 _aborted_requests 队列，下一轮 loop 迭代才 drain 做真正的 free
        self._aborted.add(req)
        self._wake.set()

    def submit(self, req):
        # req 塞 waiting + 唤醒循环
        self.waiting.append(req)
        self._wake.set()

    def start(self):
        # 挂到后台跑,立刻返回
        # asyncio.create_task(self._loop())
        self._task = asyncio.create_task(self._loop())

    def _free_kv(self, req):
        # 释放请求 kv cache
        if req.radix_tree_node:
            self.radix_tree.dec_lock_ref(req.radix_tree_node)
            req.radix_tree_node = None
        if req.kv:
            self.worker.bm.free(req.kv.block_table())
            req.kv.reset()

    def _release(self, req):
        # 释放请求
        self._free_kv(req)
        if req in self.waiting:
            self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
        req.state = State.FINISHED

    def _fail_all_inflight(self):
        # _loop 崩溃（比如 OOM）时，把所有在途请求（running + waiting 里的每一个 req）善后掉
        # 通知客户端, 释放资源, 摘出队列
        # asyncio 的技巧：queue 里可以放异常对象，stream() 的 await queue.get() 会把它在 await 点抛出来，客户端立刻收到错误而不是挂死
        e = RuntimeError("scheduler crashed")
        for req in list(self.running) + list(self.waiting):
            req.token_queue.put_nowait(e)
            self._release(req)

    async def _loop(self):
        # 死循环:都空 → await self._wake; 否则 → await self._tick()
        while True:
            try:
                if len(self.waiting) == 0 and len(self.running) == 0:
                    await self._wake.wait()
                    self._wake.clear()
                else:
                    # 这轮有没有真正 await/做事
                    # 异常会跳过 return，直接进 _loop 的 except
                    if not await self._tick():
                        await self._wake.wait()
                        self._wake.clear()
            except Exception:
                import traceback

                # 直接打 stderr，绕过 asyncio 的静默机制
                traceback.print_exc()
                print(
                    "[scheduler] _loop crashed, printing above; re-raising", flush=True
                )
                # except 块里先救人再决定死不死
                self._fail_all_inflight()
                self.fail_streak += 1
                if self.fail_streak >= 5:
                    raise

    async def _tick(self):
        # 判断 _tick 有没有做事
        made_progress = bool(self._aborted)
        # 清理断连的客户端请求
        for req in list(self._aborted):
            self._release(req)
        self._aborted.clear()

        prefills = []
        budget = self.worker.bm.num_free()
        while self.waiting:
            req = self.waiting[0]

            # 为每个请求 decode 预留块数
            reserve_blocks = sum(
                math.ceil(
                    min(
                        max(0, req.max_tokens - len(req.generated_ids)), self.decode_cap
                    )
                    * self.decode_ratio
                    / self.worker.block_size
                )
                for req in self.running
            )
            # 候选请求自己的 decode 预留(它还不在 running 里, reserve_blocks 没算它)
            own_blocks = math.ceil(
                min(max(0, req.max_tokens - len(req.generated_ids)), self.decode_cap)
                * self.decode_ratio
                / self.worker.block_size
            )

            seq = req.prompt_ids + req.generated_ids
            # 计算请求需要申请的新块数
            # 空残差截断, 防止全部匹配共享, 没有 query
            blocks, node = self.radix_tree.match_prefix(seq[:-1])
            # req.radix_matched = len(blocks)
            needed_new = (
                math.ceil(
                    (len(req.prompt_ids) + len(req.generated_ids))
                    / self.worker.block_size
                )
            ) - len(blocks)

            if (
                budget
                + self.radix_tree.evictable_blocks()
                - reserve_blocks
                - own_blocks
                < needed_new
            ):
                break
            # pin 在 gate 之后、evict 之前
            if blocks:
                self.radix_tree.inc_lock_ref(node)
                req.radix_tree_node = node
            if budget < needed_new:
                budget += self.radix_tree.evict(needed_new - budget, self.worker.bm)
                # 是否驱逐足够块数
                # 驱完还不够 → 放弃这个请求
                # 驱逐只补 prefill 缺口
                if budget < needed_new:
                    if blocks:
                        self.radix_tree.dec_lock_ref(node)
                        req.radix_tree_node = None
                    break

            # 第一次 prefill, 非被抢占, 创建 kvcache
            if req.kv is None:
                req.kv = xstar_cpp.PagedKVCache(
                    self.worker.nkv,
                    self.worker.head_dim,
                    self.worker.max_seq_len,
                    self.worker.block_size,
                    self.worker.dtype,
                    self.worker.device,
                )

            if len(blocks) > 0:
                req.kv.adopt_prefix(self.worker.bm.fork(blocks))

            prefills.append(req)
            self.running.append(req)
            self.waiting.popleft()
            budget -= needed_new

        if prefills:
            # waiting 非空且显存和可驱逐的块够新请求 → prefills = ...
            await self.worker.run_batch(prefills, is_decode=False)
            made_progress = True
        elif self.running:
            # 否则, decodes = list(running)
            decodes = []
            while self.worker.bm.num_free() < len(self.running):
                victim = self.running.pop()
                self._free_kv(victim)
                victim.state = State.WAITING
                self.waiting.appendleft(victim)
            decodes = list(self.running)
            await self.worker.run_batch(decodes, is_decode=True)
            made_progress = True

        for req in list(self.running):
            if req.state == State.FINISHED:
                self.radix_tree.insert(
                    req.prompt_ids + req.generated_ids,
                    req.kv.block_table(),
                    self.worker.bm,
                )
                self.worker.bm.free(req.kv.block_table())
                if req.radix_tree_node:
                    self.radix_tree.dec_lock_ref(req.radix_tree_node)
                    req.radix_tree_node = None
                req.kv.reset()
                self.running.remove(req)
                made_progress = True

        # 能执行到这 = 本轮干净，说明服务还活着，计数重新开始
        self.fail_streak = 0

        return made_progress
