import asyncio
import time
import httpx
from transformers import AutoTokenizer
import os
from bench.common import make_payload, stream_one

# 三档 prompt
PROMPTS = [
    ("short", "Hi"),
    (
        "medium",
        "The meaning of life is a question that has puzzled philosophers for centuries and many have tried to answer it through",
    ),
    ("long", "Once upon a time, in a land far away, " * 30),
]
MAX_TOKENS = 1  # max_tokens=1 时端到端耗时 ≈ TTFT
RUNS = 3  # 每档跑 3 次取中位数


# TTFT vs prompt 长度(单请求,无并发)
async def main():
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/Qwen2.5-0.5B"))
    timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # warmup
        await stream_one(client, make_payload("warmup", 1))
        for label, prompt in PROMPTS:
            # 真实 prompt token 数
            ptok = len(tok(prompt)["input_ids"])
            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                await stream_one(client, make_payload(prompt, MAX_TOKENS))
                times.append(time.perf_counter() - t0)
            times.sort()
            median = times[len(times) // 2]  # 取中位数
            print(
                f"prompt={label:<6}   prompt_tokens={ptok:<3}   ttft_median={median:<5.3f}s"
            )


asyncio.run(main())
