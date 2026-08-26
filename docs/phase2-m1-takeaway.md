# Phase 2 M1 能带走的东西:GPU 内存路径

> 格式对齐 [phase1-m9-takeaway.md](phase1-m9-takeaway.md) 的端到端段:每条 = 一个带具体坑的认知 + 失效场景 + 为什么。
> M1 = Phase 2 首块:CUDA 工具链接入 + Tensor GPU 内存路径(分配/搬运/释放),不算任何模型算子。验证"数据能正确进 GPU、能搬运、能回 CPU、不泄漏"。

- **三层不变量是 CUDA 集成的命门**:`cuda_allocator.h`(纯接口,无 CUDA 头,跨 g++/nvcc)只放 `void* cuda_alloc(size_t)` 这类纯 C++ 声明;`cuda_check.h`(含 `cudaError_t`/`cudaGetErrorString`,只 nvcc include)是实现工具;`.cu` 实现碰 CUDA runtime。**失效场景**:把 CHECK_CUDA 宏塞进 `cuda_allocator.h` → tensor.cpp(g++ 编)include 它时看见 `cudaError_t` 编不过(普通 g++ 没有 CUDA 头)。**为什么**:保 CXX 核心 g++ 可编、可移植,不依赖 nvcc。**判断标准可复用全项目**:函数签名有 pybind11 类型→binding 层;有 CUDA 类型→.cu 层;都没有→core 层。这条规则覆盖 tensor.cpp / cuda_allocator.cu / python_bindings.cpp 的所有归属决策。

- **cuda_free 不查返回值是刻意的非漏,不是忘了加 CHECK_CUDA**:cuda_alloc/memcpy/free_bytes 全用宏抛 runtime_error,**唯独 cuda_free 直接 `cudaFree(ptr)` 忽略返回值**。**失效场景**:给 cuda_free 加 CHECK_CUDA → 它被 `release()` 调,release 被 `~Tensor()` 调 → 析构里抛异常 → 栈展开期间第二个异常 → `std::terminate` 进程挂;且 cudaFree 失败(double-free/释放未分配)调用方也无法补救,查了返回值做不了什么。**为什么**:释放路径忽略错误跟 `std::free` 无返回值对称,是工业常见做法。注释必须讲明这个不对称是刻意的——否则下个人(或几个月后的自己)会"修"成加宏,把析构变成 terminate 炸弹。**不对称要注释,否则会被修成错**。

- **单向抛错 vs idempotent 是 M1 验收的取舍,不是对齐工业**:to_cuda 收到已是 CUDA 的 tensor 抛 RuntimeError(单向),不 no-op。**失效场景**:做成 idempotent(PyTorch `.cuda()` 那样)→ "你调 to_cuda 以为 H2D 了,但 t 已在 GPU 什么都没搬"静默错误,M1 无合法的"对已 GPU tensor 调 to_cuda"场景,严格抛错把契约显式化。**为什么 + 诚实定位**:PyTorch/SGLang(torch 栈)的 `.cuda()` 是 idempotent——单向抛错是**为 M1 抓误用 + 好测**服务的,非对齐工业。面试讲清"我选单向是为验收严格,知道工业走 idempotent,Phase 2 后期若需幂等搬运再加"。讲工业做法要诚实分层:这是设计取舍不是工业惯例。

- **反向探针:先证能红再信绿的 CUDA 落地**:round-trip 测试 green 不够——注释掉 `to_cpu` 里的 `cuda_memcpy_d2h` → 跑测试 → **必须 FAIL**(result 未拷,未初始化内存全 0,跟 x 不等)→ 还原 → PASS。**失效场景**:不拷时 result 碰巧全 0(小数组未初始化内存常零),`torch.equal` 假绿,你以为搬运对了其实没拷。**为什么**:green 证明不了搬运发生,只有"改坏就红"才证明绿是因为真逻辑——M5/M8 ablation 纪律的 CUDA 版。one-time 验证,不进测试文件(进测试会污染回归)。

- **容差臆断被打脸:cudaMemGetInfo 实测 0 波动(`feedback-industrial-claims-verify` 抓自己)**:我断言"cudaMemGetInfo 有 driver 波动,`free0==free1` 太脆要容差",实测 5 轮 alloc/free **delta 全 0**(1660 Ti 单进程 + 裸 cudaFree 同步释放 = driver 对同进程 alloc/free 确定性,精确回到原值)。**失效场景**:基于"driver 波动"的推理定容差点 → 实测 0 波动 → 容差是防御性加固防换环境假红,非"现在错";但 M7 换 pool 后此断言**失效**(pool 不真 cudaFree,回收到 free-list,free_bytes 不降,泄漏也显示 ==,得改测 pool 内部 free-list 长度)。**为什么**:不确定的要测,且测对度量——这跟 M7 的 0.1 容差被打脸同源,这条规矩这次又救了一次,而且是我抓我自己。

- **to_cuda/to_cpu 没 return(真 bug):build 警告比跑测试快**:函数签名返回 Tensor 但函数体没 `return result`。**失效场景 + 双重后果**:result 局部对象出 scope 析构 → `release()` → `cuda_free` 释放掉刚 H2D 拷进去的 GPU 内存;同时非 void 函数没 return = UB,返回栈上垃圾 Tensor,`data_` 指向已释放内存,调用方 `to_cpu` 它就是读已释放指针。**为什么**:`cmake --build` 一遍就有 `-Wreturn-type` 警告(g++ 和 nvcc 都给),build 输出里一眼可见——比写测试跑测试快。**先 build 再跑测试,build 干净是跑测试的前提**。

- **`CUDA::cudart` 不自动创建,要 `find_package(CUDAToolkit)`(我上轮臆断,探针打脸)**:我断言"启用 CUDA 语言(`project(... LANGUAGES CXX CUDA)`)就能 link `CUDA::cudart`",探针实证错——只启用 CUDA 语言不 find_package,直接 `target_link_libraries(... CUDA::cudart)` 报 "target was not found"。**失效场景**:漏 find_package → link 阶段才报错,configure 看着过。**为什么**:`CUDA::cudart` 是 `FindCUDAToolkit.cmake` 创建的 IMPORTED target,启用语言只设 `CMAKE_CUDA_COMPILER`,不建这个 target。探针造最小 CMakeLists 实测,不靠记忆——CMake 语义不确定就测,跟核 CUDA 代码同纪律。

- **raw 层只 bind 观测类,不 bind 机制类**:cuda_allocator 的 5 个 raw 函数,只 `cuda_free_bytes` 暴露给 Python,`cuda_alloc/free/memcpy_h2d/d2h` 都不 bind。**失效场景**:bind cuda_alloc → Python 能直接 alloc 裸 GPU 内存,绕过 Tensor 的 RAII,泄漏/double-free 风险;且这些被 Tensor ctor/release 内部当机制用,Python 已通过 `Tensor(...,CUDA)` 间接调,bind 是暴露实现细节。**为什么**:机制类(被 op/ctor 内部用,Python 有间接路径)不 bind;观测类(给测试/调试用,无间接路径,如读显存的 cuda_free_bytes)才 bind。避免 API 表面膨胀,只暴露 Python 真需要的。

- **CMake `CACHE STRING` 让 `-D` 覆盖生效,普通 `set` 会盖掉命令行**:`CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "..."` 而非普通 `set(... "75")`。**失效场景**:普通 set → 租卡时 `cmake -DCMAKE_CUDA_ARCHITECTURES=75;80;90` 被 CMakeLists 里那行普通 set 盖回 75 → 多卡构建静默失效,本地能跑租卡编不出 sm_80/90 的 fatbin,到租卡平台才发现。**为什么**:CMake 的普通变量优先级高于 cache 变量,普通 set 会 shadow 掉 `-D` 传的 cache 值;`CACHE STRING` 把它放进 cache,`-D` 才能覆盖。本地跑 75(1660 Ti),租卡跑 80/90(A100/H100),一编多 arch 靠这条。

- **CHECK_CUDA 宏:三次法则触发 + do-while + 不用双下划线**:4 个手写查错消费者(cuda_alloc + memcpy h2d/d2h + free_bytes)过三次法则,提宏统一。**三个坑**:(1) `do{}while(0)` 包裹——不包则 `if (cond) CHECK_CUDA(x); else ...` 的 else 错配到宏内 if,经典多语句宏坑;(2) `cudaGetErrorString` 拼 message——`throw runtime_error("CUDA error: " + cudaGetErrorString(err))` 带错误种类(oom vs invalid value),比裸 bad_alloc 好 debug,测试能 `pytest.raises(match=...)`;(3) 变量名首版用 `__cuda_err`——C++ 标准 [lex.name] 双下划线开头标识符 reserved for implementation(技术 UB),改 `cuda_err`。**为什么**:多语句宏必 do-while;throw 非 abort(alloc/copy 路径调用方可 catch,cuda_free 在析构里故不用此宏);变量名避开 reserved。宏是基础设施,docstring 讲清用法/失败行为/do-why/分层(只 .cu include)。

- **同步 cudaMemcpy,不 async——async 要 stream 抽象,裸 async 纯亏**:M1 用阻塞 `cudaMemcpy`,不用 `cudaMemcpyAsync`。**失效场景**:没 stream 时用 cudaMemcpyAsync → 跑在默认 stream,既不 overlap 计算又加同步陷阱(cudaDeviceSynchronize 漏了就是 UB),比同步纯亏。**为什么**:async 的收益是 H2D 与计算 overlap,前提是有显式 stream 管理依赖;M1 没 stream 抽象。**演进方向确定**(vLLM C++ 侧用 cudaMemcpyAsync + 显式 stream 把 H2D 和计算 overlap),Phase 5 加 stream 抽象后再切——这是确定模式,但"vLLM 具体哪行 async"没核源码,只讲模式不讲具体。

- **绿的保护伞(M1 版),测试要主动打破**:round-trip 只测"全等输入"→ 造有区分度内容(arange/randn),全 0 会藏"没拷碰巧全 0"假绿;无泄漏只测"一轮"→ 循环 N 轮(单轮 alloc/free 看不出泄漏);branches 只测"混合 CPU+CUDA"→ 拆开单独循环 CPU tensor 验"CPU 析构 free_bytes 不变"(CPU 走 std::free 不碰 GPU allocator,混在一起看不出 CPU 路径没串到 cuda_free);alloc 只测"构造不崩"→ 验元数据(nbytes/dtype/shape)对。每条都是"让本该不同的真不同,bug 藏不进去"铁律的 M1 版,同 M6 softmax 全负大幅值、M7 gate/up 顺序探针。

## 共性

Phase 2 M1 教的不是"跑通 CUDA",是"让 GPU 内存路径每一步可验证、分层可解释、臆断能被打脸"——三层不变量保可移植、不对称注释保不被修错、反向探针保 green 非碰巧、容差实测保不臆断、探针实证 CMake 语义保不靠记忆。CPU 前向作 bit-trusted oracle 一字不动,GPU 是纯增量——单变量隔离(CUDA 错 vs CPU oracle)的对拍地基。