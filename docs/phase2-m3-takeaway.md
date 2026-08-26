# Phase 2 M3 Takeaway — 手写归约 (rmsnorm + softmax)

M3 = elementwise/reduction GPU ops 的第一个 milestone,主题是**手写 block 级归约**(shared-memory tree + warp shuffle + broadcast),用两个 op 练同一套骨架的两个变体:
- **rmsnorm**:求和归约(combine = 加法),neutral = 0。
- **softmax**:**(m, l) 对归约**(combine = rescaling merge),neutral = max 下界。

两个 op 共享骨架,差别只在 combine 算子和 sentinel。下文按"可带走的东西"组织,不是流水账。

**Scope 决定:reduction only。** elementwise(silu_and_mul、residual add)不在 M3 单独写 kernel——它们该 fuse 进 M4 自写 GEMM 的 epilogue / M5 block 装配,而不是 standalone。理由:vLLM 之所以 standalone 写 `silu_and_mul` kernel(实抓 `activation_kernels.cu`:`act_and_mul_kernel`,blockIdx.x=token、grid-stride over d、独立 launch)是因为它的 GEMM 是 cuBLAS 黑盒、epilogue 插不进 activation;能自写 GEMM 的(CUTLASS epilogue / FlashInfer)走 fused。我们 M4 要自写 GEMM,standalone silu_and_mul 写了 M4 还得删。代价:M3 没练"无合作最简 kernel"(纯 grid-stride load/compute/store,无 smem/无 sync),M4 要同时学 GEMM + epilogue elementwise——负荷略高但省返工,值。

---

## 1. 骨架:block 级归约的四阶段范式(可复用到 M6 attention)

一个 block 归约一个 slice,256 线程协作:

```
阶段 1: per-thread grid-stride 累自己的 partial (寄存器)
   for (i = tid; i < size; i += blockDim.x) partial += f(x[i]);
   partial[tid] = my_partial;            // 写 shared memory
   __syncthreads();                       // [全员]

阶段 2: shared-memory tree reduce (log2(N) 轮, 每轮 __syncthreads)
   for (i = blockDim.x/2; i >= 32; i /= 2) {
       if (tid < i) partial[tid] = combine(partial[tid], partial[tid + i]);
       __syncthreads();                   // [全员, 在 if 外]
   }

阶段 3: warp shuffle 收尾 (32 → 1, 无 smem)
   if (tid < warpSize) {
       v = partial[tid];
       for (off = warpSize/2; off > 0; off /= 2)
           v = combine(v, __shfl_down_sync(0xffffffff, v, off));
       if (tid == 0) partial[0] = v;
   }
   __syncthreads();                       // [全员, 在 if 外, read 之前]

阶段 4: 全员读 partial[0], 写输出 (grid-stride)
   result = partial[0];
   for (i = tid; i < size; i += blockDim.x) out[i] = epilogue(x[i], result);
```

**可带走的不变式**:
- `__syncthreads()` 永远在**全员一致**的分支里(不在 `if (tid < i)` 内部)。CUDA 要求 block 内所有线程到达同一 barrier,否则 UB。
- 树归约读的**每个 shared slot 必须被写过**——没写就是垃圾 smem(CUDA 不清零),归约出未定义结果。neutral slot 要显式 init sentinel,不能靠"没写当 0"。
- broadcast 的 sync 要在**读 partial[0] 之前**,不是之后(race:thread 0 写 partial[0] 在 `if (tid < warpSize)` 内,其余线程在 if 外,没 barrier 就读到旧值)。
- tree 跑到 `i >= 32` 停(不是 `i > 32`)——`i > 32` 跳过 i=32 那轮,partial[32..63] 永不归约,sum 少一半,rmsnorm 输出被 √2 缩放。小 size(≤64)时 partial[32..63] 全 0,假绿。

**这是 M6 flash/paged attention 的踏脚石**——attention 的多 block merge 用的就是 softmax 的 rescaling merge,只是从单 block 升到多 block(每 block 产 (m,l),再跨 block merge)。M3 把单 block 的 primitive 练熟,M6 接多 block。

---

## 2. 数值:sentinel 跟着 combine 的单位元走,不是统一常数

这是 M3 最深的坑。归约的 neutral(空线程的初始值)必须满足 `combine(real, neutral) = real`,但不同 combine 算子的 neutral 不同:

| combine 算子 | neutral | 为什么 | 坑 |
|---|---|---|---|
| 求和 (rmsnorm) | `0` | `real + 0 = real`,`0 + 0 = 0` | 无。0 是加法单位元,天然安全。 |
| (m, l) rescaling (softmax) | `m = -FLT_MAX, l = 0` | `-FLT_MAX` 永远输给 real max(`fmaxf` 不选它),`l=0` 贡献 0 | **`-INFINITY` 会炸**:`-∞ - (-∞) = inf - inf = NaN`(IEEE 754),`0 * NaN = NaN` 传染全 block。必须用**有限地板** `-FLT_MAX`,`-FLT_MAX - (-FLT_MAX) = 0`(有限),三路 merge 都安全。 |

**可带走的判断法**:定 sentinel 前,验证三种 merge 组合在 IEEE 下都有限且正确:`(real, real)` / `(real, neutral)` / `(neutral, neutral)`。漏验 `(neutral, neutral)` 就是这轮 NaN 的根源——树归约里 neutral 之间也会 merge(dim_size < blockDim 时大量 neutral)。

**init 到首元素 + l=1 模式的 OOB 坑**(softmax):为了避开 `0 * inf = NaN`,用"读第一个元素当 m、l=1、循环从第二个开始"(FlashAttention 那套)。但 `x[threadIdx.x]` 当 `threadIdx.x >= dim_size` 时**越界读**——必须 `if (tid < dim_size)` guard。rmsnorm 没这个模式(它 0-init + grid-stride,循环守卫天然挡住越界),所以不需要 guard,也不需要 -FLT_MAX。

---

## 3. 工程:模板化消除 f32/bf16 重复

两个 op 都有 f32/bf16 双 kernel,逐行复制 ~80 行。解法:`template <typename T>` + 一个上转 helper。

**`__nv_bfloat16` 从 `float` 构造是 RNE**(cuda_bf16.h:4268 实抓:"using default round-to-nearest-even rounding mode"),`__CUDA_HOSTDEVICE__` host+device 都能用。所以:
- **读(上转)**:`to_float<T>` + `if constexpr (std::is_same_v<T, float>)` identity / `__bfloat162float`。需要分流(两种类型行为不同)。
- **写(下转)**:`(T)(expr)` 一刀切。`T=float` 时 no-op,`T=__nv_bfloat16` 时 RNE 构造,等价 `__float2bfloat16_rn`。不用分流。

**边界提醒**(不臆断):头文件承诺构造函数是 RNE,但 NaN/Inf 输入是否和 `_rn` intrinsic 逐位等价没写死。softmax 输出 [0,1] 安全;若未来 op 输出可能 NaN/Inf(未初始化的 attention 分数),用显式 `__float2bfloat16_rn` 更稳。

**helper 提工具文件**:`include/cuda/dtype_cast.h`,header-only `inline __device__`(多 .cu include 不能 multiple definition)。M4 gemm/M5 attention/mlp 全有 f32/bf16 双版本,现在建好复利。

---

## 4. 测试设计:容差纪律 + 边界覆盖 + 反向探针

### 容差(M9 纪律:先量再定,别拍)
- **f32 不 bit-exact**:GPU tree 归约 vs CPU 顺序求和,浮点加法非结合,差 ~1e-6。rmsnorm f32 实测 0(单 block 凑巧)~3.34e-6。softmax 用 `__expf`(~22 位)后比 rmsnorm 略松,实测 ~1e-5 量级。
- **bf16 比 f32 大一截**:downcast ULP dominate。rmsnorm 输出 O(1),1 ULP@bf16 ≈ 0.03125(实测命中)。softmax 输出 [0,1],1 ULP@1.0 ≈ 0.0078,实测更小。
- **诊断 rounding vs logic bug**:logic bug 给**相对误差**常数(和值成正比,如 √2 缩放 `diff/|cpu| ≈ 0.414`);rounding 给**绝对误差**被 ~1-2 ULP 封顶(值越大 ULP 越大,但 `diff/|cpu|` 反而越小)。先打 `diff` 和 `|cpu|` 看是否成正比,成正比才是 bug。

### 边界覆盖(假绿陷阱,每个都这轮踩过)
- **dim_size < blockDim**(softmax dim_size=32/2):逼 neutral 线程,暴露 sentinel / smem 垃圾 / OOB 读。**必须测到 dim_size=1, 2**——32 都不够小(smem 干净区假绿)。
- **全负 slice**(softmax):逼 max-init 是真下界。`FLT_MIN`(正 1e-38)当 max 初值会"赢"过负 real max,假绿;`-INFINITY`/`-FLT_MAX` 才对。
- **dim ≠ last / inner_size > 1**(softmax dim0/dim1_rank3):逼任意轴 3D 折叠 + `outer_idx/inner_idx` 反解 + `i*inner_size` 步长。rmsnorm 测不了(永远 last axis)。
- **grid-stride × online rescale 叠加**(softmax large_dim=4096):逼多轮 running (m,l) 状态转移,比纯加和更易错。

### 反向探针(M2 仪式)
把 kernel 的关键量钉成错误值,重跑:
- **rmsnorm**:`inv_rms → 0.0f`,输出全 0,4 个数值测试红、device_mismatch/no_leak 绿。
- **softmax**:`m_final → 0.0f`(钉 max-shift 没),输出 `expf(x-0)/l` 未归一化,6 个数值测试红、no_leak 绿。
- 探针必须**确定性失败**:别钉成可能凑巧通过的值(softmax 钉 1.0 有概率假绿,因为真 inv_rms ≈ 1;钉 0 输出有限错值,红得干净)。

---

## 5. 任意轴 softmax 的 3D 折叠(工业对齐,实抓)

PyTorch/vLLM 的 leaf softmax 都假设 contiguous,把任意秩折成 **outer × dim × inner** 三个标量,kernel 只收标量、不收 strides 数组、不收 `dim`:
- `outer_size = Π axis before dim`,`dim_size = shape[dim]`,`inner_size = Π axis after dim`。
- 元素线性偏移 `outer_idx*(dim_size*inner_size) + d*inner_size + inner_idx`。
- `blockIdx.x = slice 序号`,反解 `outer_idx = blockIdx.x / inner_size`,`inner_idx = blockIdx.x % inner_size`。

PyTorch dispatch 第一行 `input = input_.contiguous()`(SoftMax.cu:1072 实抓)——kernel 只管 contiguous,非连续由 caller `.contiguous()` 复制。我们的 `softmax.h` 契约 "x is contiguous" 同款分工,上游 attention 负责 contiguous。

vLLM 没有通用任意轴 softmax kernel(只有 MoE 用的 `topk_softmax`,绑死 last axis + topk)——通用 leaf softmax 看 PyTorch。

---

## 6. 这轮的 footgun 名单(名字即教训)

1. `FLT_MIN` = 最小**正** float(≈1e-38),不是负无穷。max-reduce 初值要用 `-INFINITY`/`-FLT_MAX`。
2. `-INFINITY - (-INFINITY) = NaN`(inf - inf,IEEE 754)。树归约 neutral 用有限地板 `-FLT_MAX`。
3. `__syncthreads()` 在发散分支 = UB。全员到达。
4. 树归约读未写的 smem slot = 垃圾。每个 slot 显式 init。
5. tree 循环 `i >= 32` 不是 `i > 32`(漏 i=32 轮,partial[32..63] 不归约,√2 缩放,小 size 假绿)。
6. broadcast sync 在 read **之前**不是之后(race)。
7. `__nv_bfloat16(float)` 是 RNE,`(T)expr` 等价 `__float2bfloat16_rn`(输出 [0,1] 安全)。
8. rescale 符号:per-thread `exp(old - m_new)` 缩小,merge `exp(较小 - 较大)` 缩小,三处方向一致。
9. init 到首元素 + l=1 模式,`x[tid]` 当 `tid >= dim_size` 越界,要 guard。
10. 测试 dim_size 覆盖到 1, 2——32 假绿。