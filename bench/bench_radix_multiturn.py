# 多轮对话模式(SGLang hicache/bench_multiturn 的单机简化: N client × R 轮, 各轮复用自己历史)。
# 每轮 1 prompt, 由"自己的历史 + 本轮新增 segment"拼成; 生成 token 文本回灌进历史。
# 命中点: 服务端 _tick 里 match_prefix(seq[:-1]) 会命中"上一轮 prompt+generated"种进树的块,
#   所以 radix on 时第 r>1 轮的 prefill = 只算本轮新增; radix off 时 = 全历史(随轮次线性涨)。
# A/B: use_radix 是 serve.app 启动参数(现为 True)。同 workload 跑两遍: 一遍 True, 一遍改 False 重启, 对比第 r 轮 TTFT.
import asyncio
import time
import httpx
from transformers import AutoTokenizer
import os
from bench.common import make_payload, stream_one

CLIENTS = 8        # 独立对话数(各聊各的, 互不共享, 测的是"单会话跨轮复用")
ROUNDS = 5         # 每 client 轮数
MAX_TOKENS = 16    # 每轮 decode 长度(生成结果回灌进历史, 让历史真实变长)
BASE = "Once upon a time, in a land far away, " * 20


async def client_conversation(client, cid):
    history = f"Client {cid} asks: " + BASE
    times = []
    for r in range(ROUNDS):
        t0 = time.perf_counter()
        out = await stream_one(client, make_payload(history, MAX_TOKENS))
        times.append(time.perf_counter() - t0)
        # 本轮生成文本回灌, 下一轮 match_prefix 才能命中"历史+生成"
        history = history + "".join(out) + f" Turn {r}: " + BASE
    return times


async def main():
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/Qwen2.5-0.5B"))
    print(f"approx_tokens/turn_prompt = {len(tok(BASE)['input_ids'])} + growing history")
    timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await stream_one(client, make_payload("warmup", 1))
        all_times = await asyncio.gather(
            *[client_conversation(client, i) for i in range(CLIENTS)]
        )
    # all_times[c][r] = client c 第 r 轮 TTFT; 转置后按轮取中位数
    per_round = []
    for r in range(ROUNDS):
        col = sorted(all_times[c][r] for c in range(CLIENTS))
        per_round.append(col[len(col) // 2])

    print("round | ttft_median(s) | vs_round1")
    for r, t in enumerate(per_round):
        print(f"  {r + 1}   | {t:<12.3f}   | {t / per_round[0]:<5.2f}x")
    print(
        "radix on: r>1 应持平或缓升(只 prefill 新段); radix off: 应随轮次线性涨(重 prefill 全历史)."
    )


asyncio.run(main())
