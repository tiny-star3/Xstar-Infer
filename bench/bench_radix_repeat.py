# 重复 prompt 模式(vLLM benchmark_prefix_caching 的 fixed 模式):
# 同一个长 prompt 连发 REPEAT 次, 第一个冷 prefill(树空, 全量, 完成后把路径种进 radix 树),
# 之后每个命中 radix, 只 prefill 被 scheduler 留作 query 的 1 个 token。
# 指标: cold_ttft(= use_radix=False 基线: 无 radix 时每次都是冷 prefill) vs warm_ttft_median。
import asyncio
import time
import httpx
from transformers import AutoTokenizer
import os
from bench.common import make_payload, stream_one

# 长共享 prompt(噪声化复读, 避免 BPE 过度压缩; 实际 token 数以运行时 tokenize 为准)
SHARED = (
    "In the annals of computational history few artifacts have reshaped the field as profoundly as the transformer, whose self-attention mechanism replaced the sequential recurrence of prior architectures and unlocked a new regime of parallel scalable learning for language and vision alike. "
    * 30
)
REPEAT = 32  # 连发次数: 第 1 个冷, 后 31 个热
MAX_TOKENS = 1  # max_tokens=1 时端到端耗时 ≈ TTFT, 只隔离 prefill(radix 省的正是 prefill)
RUNS = 3  # warm 段取 RUNS 个 batch 的中位数, 抗抖动


async def timed_stream_one(client, prompt):
    t0 = time.perf_counter()
    await stream_one(client, make_payload(prompt, MAX_TOKENS))
    return time.perf_counter() - t0


async def main():
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/Qwen2.5-0.5B"))
    ptok = len(tok(SHARED)["input_ids"])
    timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # warmup(不同 prompt, 不进 SHARED 的 prefix, 不污染命中)
        await stream_one(client, make_payload("warmup", MAX_TOKENS))

        # 冷请求: 树空 → 全量 prefill, 完成后把 SHARED 的块种进 radix 树
        cold = await timed_stream_one(client, SHARED)

        # 热请求: 命中 radix, 只 prefill 被留作 query 的那段
        warm = []
        for _ in range(RUNS):
            times = await asyncio.gather(
                *[timed_stream_one(client, SHARED) for _ in range(REPEAT)]
            )
            t = sorted(times)
            warm.append(t[len(t) // 2])
        warm.sort()
        warm_median = warm[len(warm) // 2]

    print(f"prompt_tokens={ptok:<5}   repeat={REPEAT:<3}")
    print(
        f"cold_ttft={cold:<6.3f}s   warm_ttft_median={warm_median:<6.3f}s   "
        f"speedup={cold / warm_median:<5.2f}x"
    )


asyncio.run(main())
