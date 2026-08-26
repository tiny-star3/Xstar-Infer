# Phase 3 M3 Takeaway — Continuous Batching + Recompute 抢占 + Bench

M3 = Phase3 的收口 milestone,主题是**多请求并发推理 + 显存压力下抢占式调度 + 性能基准**。在 M1(多请求 kernel:per-warp-one-seq decode + 3D-varlen prefill)和 M2(FastAPI 单请求流式)之上,把 forward 调用权从"每请求自调"收归调度器,实现 continuous batching,并补上 vLLM v0.6.0 式的 Recompute 抢占与 httpx 异步 bench。

下文按"可带走的东西"组织,不是流水账。每条尽量挂上和 Phase1/M1/M2 的同构或承接。

---

## 1. 调度器统一调用取代"每请求自调 forward" = M3 的结构地基

M2 是 `stream()` 里每请求自己 `await run_in_threadpool(forward)`;M3 把 forward 调用权收归 `_tick`,**一个 `run_batch(decodes)` 喂 N 个请求**。请求只往 `waiting` submit,不自己驱动 forward——这是从"每请求一个 coroutine 跑 forward"到"一个调度器 coroutine 跑整批"的范式切换,continuous batching 的前提。

承接 M2 第 6 点:M2 走 `qwen2_forward_multi`(num_seqs=1 退化)而非 `qwen2_forward_paged`,M1 的多请求 kernel **就是为 M3 写的**。M3 把 num_seqs 从 1 拉到 N,kernel 零改——跟 M1"per-warp-one-seq 为 splitKV 铺路"、M2"用未来接口跑当前阶段"是同一类前瞻取舍,M3 是那个"未来"落地。

## 2. Recompute 抢占的三段职责切分:reset 不碰 bm 是设计不是偷懒

`PagedKVCache::reset()` 只清自己的状态(`cursor_=0` / `block_table_.clear()` / `cuda_free(d_block_table_)`),**故意不碰 bm**(头文件 docstring 写死 "Does NOT touch bm (no reference by design)")。bm 的 block 释放是 scheduler 侧 `bm.free(victim.kv.block_table())` 单独做。重 prefill 时 worker 喂 `prompt_ids + generated_ids`(完整,不是只 prompt)复现 KV。

why 切三段:PagedKVCache 不持有 bm 引用 → reset 不会误释放共享池里的 block(否则多请求共用一个 bm 会互相踩)。这是"职责边界要硬"——跟 M2 第 4 点"跨语言边界类型不能混用"同源:**边界处的所有权契约要精确,reset 释放自己拥有的(d_block_table)、不释放自己不拥有的(bm block)**。

## 3. 抢占触发判据的保守性 + pool 大小张力(两个 bench 口径别混)

触发条件 `num_free() < len(running)`(vLLM v0.6.0 同款保守判据,不算 cursor 边界)。但这判据意味着**只有池接近耗尽才抢**——pool=16384(`*8`)时 24 并发 ×1000 token 才占 900 block,free 最低 536 >> 24,**抢占永不触发**;pool=150 时 8 并发 ~273 token 才触发。

张力:pool 大 → 抢占不可达但 batch 能堆大(测吞吐口径);pool 小 → 能触发抢占但 prefill 长 prompt 易直接 OOM 抛 `insufficient free blocks`(不是优雅抢占)。**标准 bench 用大 pool,抢占验证用小 pool,两者不能混**。这跟 M2 第 2 点"bm 全局为 M3 预留"的前瞻一脉相承,但 M3 暴露了"预留多大"本身就是个设计张力,不是越大越好。

## 4. binding 三处同改,漏一处 → 后台 task 静默崩 → 表象是 hang 不是报错

`reset()` 在 `paged_kv_cache.h` 声明、`.cpp` 实现都有,但 `python_bindings.cpp` 漏了 `.def("reset", ...)` → Python 侧 `AttributeError: 'PagedKVCache' has no attribute 'reset'`。这个 AttributeError 在 `_tick` 末尾 FINISHED 清理时抛,**崩的是后台 `_task` 协程,没人接异常**,表象是"请问"请求 hang 住(不是报错退出)。

教训两条:
- (a) pybind 绑定是**三文件契约**(h/cpp/bindings),漏一个就崩。跟 M1"容差 probe 不臆断"、M2 第 5 点"binding 契约查实际调用别看签名"同源——静态信息(声明在)不等于运行时可用(绑了没)。
- (b) **后台 task 崩溃的表象是静默 hang 不是 crash**——新坑。排查时容易往"死锁/卡 GPU"想,实际是 task 已死。诊断要点:后台 task 一定要能让人看到它的异常(日志 / 或 `_loop` 包 try 接住),否则就是隐形的。

## 5. bench 读出两个瓶颈:TTFT 固定开销平台 + 吞吐拐点 = memory-bound 显现

**TTFT**:短 prompt(1 token vs 21 token)中位数 0.123s vs 0.118s **几乎相同甚至反序**,长 prompt(331 token)才跳到 0.817s → 短 prompt 被 **~120ms 固定开销平台主导**(24 层 forward 的 kernel launch + Python→C++ 跨边界 + `run_in_threadpool` 调度 + 网络),prefill 计算量太小被淹没。

**吞吐**:1→32 近线性 20×(8.9→181.5 tok/s),32→128 边际骤降(+8.6%/+9.4%);TPOT 32→128 每翻倍近翻倍(176→325→594ms)。**拐点 batch=32 = 权重访存摊薄红利吃完、KV 访存成主导的临界**,memory-bound 教科书曲线实测落地。

承接 M1 probe 分层法,但 **probe 的对象从"单 op 数值"变成"bench 曲线形状"**——不是打印一层 tensor,是看曲线哪里不随长度变(固定开销)、哪里斜率变(拐点)。同样是"分层定位瓶颈",M3 这层是宏观的。

## 6. bench harness 的 client 边界契约 —— 边界精确落到 client 侧

四个踩坑:
- (a) httpx 0.28 `Timeout(connect=, read=)` 不给 default 直接报错,要么全四参要么给 default。
- (b) `httpx.Limits(max_connections=200)` 必设,默认连接池上限低,batch=128 测的是排队不是吞吐。
- (c) `gather(*[stream_one(...) for _ in range(B)])` 要一次性构造 B 个 coroutine 并发,写成循环 `await` 就退化串行(测出来和 batch=1 一样)。
- (d) SSE 的 `data: {}\n\n` 双换行,`aiter_lines()` 按单 `\n` 切出空行要跳,`[DONE]` 是结束信号不是 token 要排除。

承接 M2 第 4 点"跨语言边界类型不能混用"——M2 的边界在 Python↔C++(枚举 vs torch device),M3 的边界在 **Python↔HTTP/SSE**(str vs bytes、连接池语义、并发原语)。同一类教训:**每个边界都有自己的契约,不能靠语义猜,要查实际行为**。

## 7. bit-exact 边界升级:从 M1"流畅≠正确"到 M3"bit-exact≠路径等价"

抢占 recompute 验证:victim 重 prefill 输出 vs 隔离单跑 **400 token 逐个 bit-exact**。但严格说,重 prefill 走 **varlen 路径**(一次性算 prompt+generated 的 KV),原 decode 走**增量路径**(逐 token 算),两条路在 bf16 下理论 8-ulp 差异(M1 实证)。bit-exact 是因为**误差没翻转任何一步 argmax**,不是两条数值路径等价。

这是 M1 第 7 点"流畅输出对 8-ulp 平局翻转失明"的**判据升级**:M1 是"文本流畅 ≠ 数值对",M3 是"bit-exact ≠ 路径等价"。两个方向都是**强判据被弱证据满足时的警惕**——M1 警惕"流畅骗你",M3 警惕"bit-exact 骗你以为路径等价"。贪心确定路径下 bit-exact 成立,一旦上 sampling(temperature>0)或跨零临界 argmax,这个 bit-exact 就可能破——这是 recompute 进真实 sampling 服务的命门。

---

## 留到 Phase4 的债

- **bench 数不对账**:215.6 tok/s"接近饱和"是看曲线形状(边际骤降)推断,**没和 GPU TFLOPS/带宽理论 tok/s 上界对账**。同 M2 第 7 点"判据在没执行到底"。Phase4 bench 时补理论对账。
- **orphan abort 没做**:client 断连不释放 block,scheduler 继续跑幽灵请求至 EOS/max_tokens。真实 serving 缺口。Phase4 一起改(`stream` finally 标 aborted + `_wake.set` + `_tick` 开头收尸)。
- **关闭不干净**:`scheduler._task.cancel()` 裸调没 await,`_loop` 没 try 兜 CancelledError。
- **bench_async 没取中位数**:大 batch 抖动。
- **max_tokens 语义**:prefill 首个 token 计入 generated_ids 配额(用户有意?未确认)。

## bench 硬数据(httpx 异步,大 pool=16384,Qwen2.5-0.5B/6GB 卡/贪心)

吞吐 vs batch(max_tokens=64):

| batch | tokens | elapsed | throughput | TPOT |
|-------|--------|---------|------------|------|
| 1 | 64 | 7.20s | 8.9 tok/s | 112.5 ms |
| 2 | 128 | 7.12s | 18.0 tok/s | 111.3 ms |
| 4 | 256 | 7.40s | 34.6 tok/s | 115.7 ms |
| 8 | 512 | 7.90s | 64.8 tok/s | 123.4 ms |
| 16 | 1024 | 9.07s | 112.9 tok/s | 141.7 ms |
| 32 | 2048 | 11.28s | 181.5 tok/s | 176.3 ms |
| 64 | 4096 | 20.78s | 197.1 tok/s | 324.8 ms |
| 128 | 8192 | 38.00s | 215.6 tok/s | 593.8 ms |

TTFT vs prompt(max_tokens=1,3 次中位数):1 token→0.123s / 21 token→0.118s / 331 token→0.817s。

抢占 recompute(pool=150):8 并发续写 prompt × max_tokens=400,LIFO 队尾 victim 被 reset 重 prefill,输出 vs 隔离单跑 400 token 逐个 bit-exact。
