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


class NoRadixTree:
    # 空对象桩
    # 方法面与 RadixTree pybind 绑定一一对齐，签名改动必须两边同步

    def match_prefix(self, tokens):
        # 0 token 命中 → needed_new = 全量 → 退化成老路径
        return (0, None)

    def insert(self, tokens, block_table):
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
    def __init__(self, worker, use_radix=False):
        self.worker = worker
        self.waiting = deque()
        self.running = deque()
        self._wake = asyncio.Event()  # 唤醒空转的循环
        self._task = None  # 后台自己跑的协程
        self.radix_tree = (
            xstar_cpp.RadixTree(worker.block_size) if use_radix else NoRadixTree()
        )

    def submit(self, req):
        # req 塞 waiting + 唤醒循环
        self.waiting.append(req)
        self._wake.set()

    def start(self):
        # 挂到后台跑,立刻返回
        # asyncio.create_task(self._loop())
        self._task = asyncio.create_task(self._loop())

    async def _loop(self):
        # 死循环:都空 → await self._wake; 否则 → await self._tick()
        while True:
            if len(self.waiting) == 0 and len(self.running) == 0:
                await self._wake.wait()
                self._wake.clear()
            else:
                await self._tick()

    async def _tick(self):
        prefills = []
        decodes = []
        matched_len, node = self.radix_tree.match_prefix(
            self.waiting[0].prompt_ids + self.waiting[0].generated_ids
        )
        needed_new = (
            math.ceil(
                len(self.waiting[0].prompt_ids) + len(self.waiting[0].generated_ids)
            )
            / self.worker.block_size
        ) - matched_len / self.worker.block_size
        if (
            len(self.waiting) != 0
            and self.worker.bm.num_free() + self.radix_tree.lru_size() >= needed_new
        ):
            # waiting 非空且显存够 → prefills = ...; decodes = []
            need_blocks = needed_new - self.worker.bm.num_free()
            if need_blocks > 0:
                num_evict = self.radix_tree.evict(need_blocks, self.bm)
            if num_evict == need_blocks:
                self.bm.fork()
            if self.waiting[0].kv is None:
                self.waiting[0].kv = xstar_cpp.PagedKVCache(
                    self.worker.nkv,
                    self.worker.head_dim,
                    self.worker.max_seq_len,
                    self.worker.block_size,
                    self.worker.dtype,
                    self.worker.device,
                )
            prefills.append(self.waiting[0])
            self.running.append(self.waiting[0])
            self.waiting.popleft()
            await self.worker.run_batch(prefills, is_decode=False)
        else:
            # 否则, prefills = []; decodes = list(running)
            while self.worker.bm.num_free() < len(self.running):
                victim = self.running.pop()
                self.worker.bm.free(victim.kv.block_table())
                victim.kv.reset()
                victim.state = State.WAITING
                self.waiting.appendleft(victim)
            decodes = list(self.running)
            await self.worker.run_batch(decodes, is_decode=True)

        for req in list(self.running):
            if req.state == State.FINISHED:
                self.worker.bm.free(req.kv.block_table())
                req.kv.reset()
                self.running.remove(req)
