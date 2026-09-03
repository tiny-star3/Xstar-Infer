# Phase 4 M3 能带走的东西：splitKV decode 微优化

> 格式对齐 `phase4-m1/m2-takeaway.md`：每条 = 一个带具体坑的认知 + 失效场景 + 为什么。
> M3 = Phase4 第三块：FlashDecoding 风格 splitKV decode（`paged_attention_decode_split.cu`：per-split partial + LSE merge 两 kernel）+ kernel 级微 bench（`bench/bench_splitkv.py`）+ 计数探针。核心不是 kernel 本身，是"什么时候值得 split"的判据链。

---

**1. split 的收益阈值是 bench 出来的，不是拍的：只在 ≥8K 赚，≤2048 反而亏**

- 微 bench（`torch.cuda.Event` 直测 forward，J=10，Qwen2.5-0.5B / 6GB 卡，dummy arange）：batch=1 下 512→1.05x / 2048→1.08x / 8192→**1.52x** / 32000→**1.53x**（S=16）
- ≤2048 在 1.0 附近**对称抖动**：每层每步多一次 kernel launch + 一次 partial buffer 的 cuda_alloc，小 seq 时固定开销吃掉并行收益——split 轴并行度省下的访存时间不够付 launch 税
- 落地：`SPLIT_KV_THRESHOLD` 从 512 抬到 4096（`paged_attention.cpp`）。第一版直觉值 512 太低，被 bench 打脸——阈值必须被数据支撑，不是"分得越细越并行越好"

**2. 两 kernel 结构：per-split partial（带 m/l）+ LSE merge——online softmax 从"跨 block"搬到"跨 split"**

- 每 split 算自己的 running-max m 和 sum-exp l；merge kernel 做 max 归 max、`exp(logit − m_max)` rescale 后加权求和
- 承接 M8 FA2：同一套 online softmax 数学，作用对象从"KV cache 分块"换成"seq_len 切分"，不是新数学。FlashDecoding 与 FA2 的分工：FA2 省的是 KV 读写（一读一算），splitKV 补的是低 batch 下 seq 轴并行度

**3. cap=16 是已测上限，S=32 没测过——常量定死"已验证"值**

- S=16 压 S=8（8192：1.36→1.52；32000：1.45→1.53），但 S=32 未测 → `MAX_SPLITS=16` 定死在政策常量
- "想抬先补 bench"——跟"只讲确定的"同一条：没测过的数字不进代码注释、不进简历

**4. batch=4 收益衰减（8192 只有 1.21x）：splitKV 是低 batch 长 context 的优化，不是普适的**

- batch 轴已喂满 SM 时，split 轴抢不到更多带宽，只剩额外 launch/alloc 开销
- 推论：decode 场景 bench 前先看 batch——batch 大时 splitKV 收益本来就小，别拿 batch=1 的 1.5x 当普适结论
- batch=4×32000 在 6GB 卡 OOM（prefill 激活超池）——**容量边界，不是 kernel bug**，别记成正确性问题

**5. 计数探针：keys_walked 不漂 → 切分是纯重划分，无重叠无漏格**

- device atomic 计 keys_walked，恒 = num_heads × Σ seq_k = 14×678 = **9492**，S∈{2,8,16} 都一样；测完探针已回滚
- allclose GREEN 只证"没算错"，不证"split 真发生且走查完整"——走查不变量断言比 argmax 对拍硬。承接 M8 skip 计数探针、"优化要计数探针"feedback：数值 GREEN 是必要非充分

**6. 诚实边界：kernel 级 headline 与系统级 headline 分开标，两个量级不能混**

- 这是 **decode kernel 微优化**，非整系统吞吐。简历 bullet："手写 FlashDecoding 风格 splitKV decode（两 kernel：per-split partial + LSE merge，running-max 在线合并），8K+ context decode 延迟 −33%（1.5x）"
- Phase4 的整系统 headline = **RadixAttention 前缀复用**：repeat 3.1x / multiturn 第 5 轮 4.51x（见 bench/README.md）
- 混着说会被抓："1.5x 是哪个口径"答不上来就是减分项

**7. 微 bench 方法：Event 直测 forward，不端到端**

- kernel 级收益用 kernel 级口径；端到端会混进 ~120ms 固定开销平台（Phase3 TTFT 结论），把 1.5x 淹掉
- 承接 M1 probe 分层法：测哪层的东西用哪层的口径，跨层测量会把信号淹没在别的层的噪声里

---

## 共性

M3 教的是**微优化的判据链**：微 bench（值不值，阈值从数据来）→ 计数探针（真发生 + 走查完整）→ 常量定死已验证值（没测过的不写）。跟 M8 skip 探针、Phase3 理论对账同一条链，这期补的是"阈值"这一环——第一版拍的 512 被 bench 打脸是最直接的证据。另一条是**口径纪律**：kernel 级与系统级 headline 分开标、容量问题与正确性问题分开记，"1.5x" 必须能答上"哪个口径、什么条件"。
