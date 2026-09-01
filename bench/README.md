# Benchmarks

Xstar-Infer 推理引擎的性能基准。所有 bench 针对单机服务 `serve.app`(FastAPI + uvicorn),
模型 Qwen2.5-0.5B(bf16, GQA, 24 层),贪心 argmax 解码(无 sampling)。

## 前置

启动服务(在 repo 根):

```bash
.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

**KV cache pool 设置**(`serve/app.py:51` 的 `num_blocks`):
- `bench_async` / `bench_ttft` → 用 `* 8`(= 16384 block,~3.2GB KV 池)
  - 6GB 卡上这是"尽量占满显存又留够激活余量"的配置(总占 ~5GB,留 ~1GB 给大 batch 激活)
  - 大 pool 才能堆大 batch,吞吐才是真实负载口径
- 抢占验证(见附录)→ 临时改 `150`,验证完改回 `* 8`

跑前先 warmup 一个请求排除 CUDA 冷启动(脚本内置)。

## bench_async:吞吐 vs batch

**测什么**:continuous batching 下,吞吐(tokens/s)和 per-token 延迟(TPOT)
随并发 batch size 的扩展性。验证"batch 把 GPU 从单请求饿死喂到接近算力/带宽上限"。

**跑法**:

```bash
python -m bench.bench_async   # 或 python bench/bench_async.py
```

并发扫 `batch = [1, 2, 4, 8, 16, 32, 64, 128]`,每路 max_tokens=64,续写类 prompt。
用 httpx 异步 `AsyncClient` 长连接复用 + `httpx.Limits(max_connections=200)`(解除默认连接上限,
否则 batch=128 测的是排队不是吞吐)。

**结果**(httpx 异步,3 次未取中位数,单跑):

| batch | tokens | elapsed | throughput | TPOT |
|-------|--------|---------|------------|------|
| 1     | 64     | 7.20s   | 8.9 tok/s  | 112.5 ms |
| 2     | 128    | 7.12s   | 18.0 tok/s | 111.3 ms |
| 4     | 256    | 7.40s   | 34.6 tok/s | 115.7 ms |
| 8     | 512    | 7.90s   | 64.8 tok/s | 123.4 ms |
| 16    | 1024   | 9.07s   | 112.9 tok/s| 141.7 ms |
| 32    | 2048   | 11.28s  | 181.5 tok/s| 176.3 ms |
| 64    | 4096   | 20.78s  | 197.1 tok/s| 324.8 ms |
| 128   | 8192   | 38.00s  | 215.6 tok/s| 593.8 ms |

**判读**:
- 吞吐随 batch 从 8.9(batch=1)扩展至 **215.6 tok/s(batch=128,24×)**
- **拐点在 batch=32**:1→32 近线性(20×);32→64 只 +8.6%,64→128 只 +9.4% → GPU 接近饱和,
  215.6 tok/s 是这台 6GB 卡上 0.5B 模型的吞吐天花板
- **TPOT 在 batch=32 后陡升**:1→32 缓升(112→176 ms,1.6×,权重访存被 batch 摊薄);
  32→64 跳至 325 ms,64→128 跳至 594 ms(每翻倍 batch TPOT 近翻倍)→ KV cache 访存成主导,
  decode memory-bound 特征显现
- **batch=32 是吞吐/延迟甜点**:181.5 tok/s + 176ms TPOT;其后是"用更差延迟换更多吞吐",过甜点区
- batch=2 TPOT(111)略低于 batch=1(112):batch=1 时 GPU 利用率极低,固定开销(kernel launch/调度)占比大,batch=2 摊薄

## bench_ttft:TTFT vs prompt 长度

**测什么**:prefill 阶段延迟(compute-bound)随 prompt 长度的趋势。TTFT = 发请求到收第一个 token。

**跑法**:

```bash
python -m bench.bench_ttft
```

单请求(无并发),max_tokens=1(此时端到端耗时 ≈ TTFT),三档 prompt 长度,
每档跑 3 次取中位数。用 tokenizer 数 prompt 真实 token 数。

**结果**:

| prompt | prompt tokens | TTFT(中位数) |
|--------|---------------|--------------|
| "Hi"   | 1             | 0.123s       |
| 中等句 | 21            | 0.118s       |
| 长句   | 331           | 0.817s       |

**判读**:
- 短 prompt(1 token)与中 prompt(21 token)TTFT 几乎相同(0.123 vs 0.118),中 prompt 甚至略低
- **原因**:短 prompt 的 prefill 计算量对 0.5B 模型过小,被 **~120ms 固定开销主导**
  (24 层 forward 的 kernel launch + Python→C++ 跨边界 + `run_in_threadpool` 调度 + 网络),
  长度差异被淹没
- **长 prompt(331 token)** 计算量盖过固定开销,TTFT 升至 0.817s → compute-bound 才显现
- 结论:TTFT 在短 prompt 区间 ≈ 固定开销平台(~120ms,与长度无关);长 prompt 区间随长度涨。
  固定开销是后续优化点(减少跨语言 round-trip、kernel fusion)

## 测量口径与 caveat

- **TTFT 测法**:`max_tokens=1` 时的端到端耗时近似 TTFT。httpx 流式下,
  `stream_one` 收完 1 个 token(遇 `[DONE]` 退出)即返回,故其耗时 ≈ TTFT。
- **TPOT 定义**:端到端 per-token(`elapsed / 平均实际 token`),含网络/调度开销,
  非纯 GPU decode 时间。并发下 TPOT = 整批每步墙钟(整批每步出 B 个 token,
  `elapsed / (total/B)` = 单步时间)。请求提前自然 EOS 时用实际 token 数,不用 max_tokens。
- **未取中位数**:bench_async 每档单跑,未多次取中位数,数字有抖动(尤其大 batch 跑久)。
  复现建议每档跑 3 次取中位数,同 bench_ttft。
- **贪心解码**:无 sampling,输出确定路径,延迟波动主要来自排队/batch 而非 sampling 抖动。

## 附录:抢占 + Recompute 正确性(手动验证,非自动化 bench)

前置:pool 改 `150`(人为制造显存压力),服务重启。

8 并发续写类 prompt(不易自然 EOS),max_tokens=400。抢占触发条件
`num_free() < len(running)`(池耗尽),victim = `running.pop()`(LIFO 队尾)。
被抢 victim 被 `reset()` + 放回 waiting 头,下个 tick 用 `prompt_ids + generated_ids` re-prefill。

**结果**:队尾请求被抢占(进度落后 316 vs 其他 400,最终补齐到 400),
其完整输出与**隔离单跑同一 prompt**(从未被抢占)**逐 token bit-exact 一致(400 token)**。

**结论**:Recompute 模式下,reset 后用 prompt+generated 重算的 KV 复现了原 decode 累积的 KV,
被抢占点之后 argmax 无漂移。

**诚实边界**:bit-exact 是在贪心确定路径下成立。严格说 re-prefill 走 varlen 路径、原路径走
增量 decode,两条路在 bf16 下理论上存在 8-ulp 量级差异;bit-exact 是因为误差未翻转 argmax,
不是数值路径完全等价。验证完 pool 改回 `* 8`。

## bench_radix:前缀缓存(RadixAttention)效率

两个 workload,照 vLLM `benchmark_prefix_caching.py` 与 SGLang `hicache/bench_multiturn.py` 的工业口径:
**同一 workload 下 `use_radix` 开/关各跑一遍做 A/B**,不是手工拼 hit/miss 两批。开关在
`serve/app.py:58`(`Scheduler(use_radix=...)`);off 半场服务端 `NoRadixTree` 桩自动退化(radix-off == 无缓存)。

### bench_radix_repeat:共享 system prompt

同一长 prompt 连发 `REPEAT=32` 次(1411 token):第 1 个冷(全量 prefill,完成后路径种进 radix 树),
后 31 个热(命中,只 prefill 被 scheduler 留作 query 的 1 token)。取 warm 段 batch 中位数对 cold 得 speedup。

**结果**(2026-09-01,6GB 卡,0.5B,`MAX_TOKENS=1` 时端到端 ≈ TTFT):

| 口径 | 耗时 |
|------|------|
| cold(radix-off 基线 = 每次全量 prefill) | 3.49s |
| warm(命中,只 prefill 1 token)         | 1.19s |
| speedup                               | **3.1x** |

**判读**:radix 省的是 prefill 计算,TTFT 是锐利指标。cold 含 ~120ms 固定开销平台,SHARED 越长
(全量 prefill 越贵) speedup 越接近真实上限;warm 被 120ms 平台主导,是下界。

### bench_radix_multiturn:多轮对话逐轮复用

N client × R 轮,每轮 prompt = 自己历史 + 新增段,生成文本回灌,逐轮测 TTFT 中位数。
on/off 两半同参数对齐(8 client、BASE=221 token、`MAX_TOKENS=16`)。

**结果**(2026-09-01,对齐后):

| 轮 | radix ON | radix OFF | off/on |
|----|----------|-----------|--------|
| 1  | 5.94s    | 5.88s     | ~1.0x  |
| 2  | 6.07s    | 10.18s    | 1.68x  |
| 3  | 6.36s    | 14.94s    | 2.35x  |
| 4  | 6.74s    | 24.69s    | 3.66x  |
| 5  | 6.87s    | 30.96s    | **4.51x** |

**判读**:
- round-1 两边同为 ~5.9s = 自校验(第 1 轮无论开关都是冷跑,harness 测的是同一件事)
- on 只缓升到 1.16x(decode 每轮读更长的 KV,memory-bound 主导);off 暴涨到 5.27x(每轮重 prefill 全历史)
- 第 5 轮 off/on = **4.51x**,比 repeat 的 3.1x 更猛——因为累积了 5 轮历史,这才是 radix 的真实量级

### 复现协议

```bash
# radix ON(默认)
.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8000
.venv/bin/python -m bench.bench_radix_repeat
.venv/bin/python -m bench.bench_radix_multiturn

# radix OFF:改 app.py:58 = False → 重启 → 跑同样的 bench_radix_multiturn → 改回 True 再重启
```

**坑(必读)**:off 半场必须干净重启 + `kill -9` 清掉旧 client。后台 bench 被 SIGTERM 杀会留孤儿请求
挂在 scheduler.running、占着 block;pool 耗尽 → scheduler `_loop` 死于 OOM 后 re-raise → **loop 已死
但 HTTP 仍 listen**,后续请求全部挂起(phase3 债务"orphan abort 关闭不干净",也是三种进程死里最隐蔽的)。

