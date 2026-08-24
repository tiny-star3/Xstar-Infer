import asyncio
import time
import httpx
from bench.common import make_payload, stream_one

PROMPT = "The story continues as the hero ventured deeper into the ancient forest where mysteries"
MAX_TOKENS = 64
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


# 吞吐 vs batch 曲线
async def main():
    # 解除连接池上限
    limits = httpx.Limits(max_connections=200)
    # read=None 流式不超时
    timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # warmup 丢掉结果
        await stream_one(client, make_payload("warmup", 1))
        for B in BATCHES:
            t0 = time.perf_counter()
            results = await asyncio.gather(
                *[  # 一次性并发 B 个
                    stream_one(client, make_payload(PROMPT, MAX_TOKENS))
                    for _ in range(B)
                ]
            )
            elapsed = time.perf_counter() - t0
            # 实际 token 数,防自然 EOS
            total = sum(len(r) for r in results)
            tps = total / elapsed
            # 端到端 per-token,用实际平均
            tpot = elapsed * 1000 / (total / B)
            print(
                f"batch={B:<3}   tokens={total:<4}   elapsed={elapsed:<5.2f}s   tps={tps:<5.1f}   tpot={tpot:<5.1f}ms"
            )


asyncio.run(main())
