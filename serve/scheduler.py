import asyncio
import enum
from collections import deque
import math


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


class Scheduler:
    def __init__(self, worker):
        self.worker = worker
        self.waiting = deque()
        self.running = deque()
        self._wake = asyncio.Event()  # 唤醒空转的循环
        self._task = None  # 后台自己跑的协程

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
        if len(self.waiting) != 0 and self.worker.bm.num_free() >= math.ceil(
            (len(self.waiting[0].prompt_ids) + len(self.waiting[0].generated_ids))
            / self.worker.block_size
        ):
            # waiting 非空且显存够 → prefills = ...; decodes = []
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
