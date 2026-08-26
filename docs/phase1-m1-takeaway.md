# Phase 1 M1 能带走的东西

## 一、M1 的目标与验收判据

**目标:** 产出一个数值可信、内存安全的 C++ Tensor 运行时 + mmap 零拷贝权重加载,作为 Phase 1-4(C++ 前向对拍 / PagedAttention / FastAPI / Continuous Batching)的地基。M1 不算任何模型算子,只验证"数据能正确进来、能零拷贝映射、能正确出去"。

**核心判据(全部已验证):**

1. **Tensor 基建正确(强):** `from_numpy → Tensor → to_numpy` 往返一致;owned 内存独立(改返回数组不影响原 Tensor);bf16 的 `nbytes == numel×2`;`[8,896]` 等非平凡形状构造正确;
2. **bfloat16 RNE 正确(强):** round-to-nearest-even 五点(tie-even→0x3f80、tie-odd→0x3f82、round-up、truncate)全部验对,bias trick `0x7FFF + lsb` 机制正确;
3. **mmap 零拷贝真实(强):** `make_weight_view` 造的视图指针直接落在 mmap 区,不 alloc 不 memcpy;别名法实证"改底层文件后同一视图实时反映新内容";
4. **防御性校验(强):** 越界(nbytes>文件)、未对齐(offset 不整除 dtype_size)均抛异常,越界校验用减法防整数溢出;
5. **内存安全(强):** ASan 验 owned/view 两条析构路径无 double-free、无 leak;UBSan 验无 fall-off UB。

**验收方式:** `pytest tests/test_cpp_tensor.py` 10 项全绿(基建 3 + mmap 7),0.22s,可一键回归。

## 二、设计决策及其依据

### 2.1 RAII 三层守卫,析构绝不抛异常

资源分三层,各管一种:`UniqueFd`(fd)→ `MMapFile`(mmap 映射)→ `Tensor`(buffer)。每层构造即持有、析构即释放,非拷贝、可移动。

**析构绝不抛异常的依据:** 若对象恰在栈展开(另一个异常正向上传播)过程中被析构,析构再抛第二个异常 → C++ 运行时 `std::terminate`,进程挂,无恢复机会。故 `~MMapFile` 里 `munmap` 失败只能 swallow + log,不能抛。`~UniqueFd` 同理——它专门用于栈展开路径,`close` 失败必须吞掉。

### 2.2 move 传播 `owns_data_`,两条析构路径行为不同

Tensor 有两条构造路径:owned(`aligned_alloc`,析构 `free`)与 view(指针指向外部 mmap,析构不 free)。区分靠 `owns_data_` 标志。

**关键不变量:** move ctor / move assign 都把 `owns_data_` 跟着 `data_` 一起搬。否则"把一个 view move 给 owned"会变成"接收方析构时 free 掉 mmap 区"→ double-free / 误释放。`release()` 用 `if (data_ && owns_data_)` 守卫,view 析构不碰 mmap。

**两条路径行为对照(已各写一个测试钉死):**
- owned 路径:`to_numpy` 每次拷贝,改返回数组不影响原 Tensor(独立);
- view 路径:别名共享,改底层文件视图实时反映(零拷贝)。

### 2.3 半构造不泄漏:清理 close 不检验,业务 close 检验

构造函数里每拿一个资源,后面所有可能抛异常的路径都要保证已拿资源被释放。用 `UniqueFd` 守卫包 fd——清理路径靠守卫析构自动 best-effort close,业务路径 `release()` 取出 fd 手动 close + 检验。

**为什么清理路径的 close 不检验:** 你已经在为别的原因抛异常,close 再失败只能吞掉(吞进守卫析构),不可能再抛第二个。**只有成功路径上那个 close 才值得检验。** 这个区分由 `unique_fd` 机械化:清理靠守卫 dtor 吞,业务靠 `release()` 后手动检——再写不出 fd 泄漏。

### 2.4 mmap 零拷贝:为什么 mmap,为什么 MAP_SHARED

**为什么 mmap 而非 read:** 省一次大拷贝(权重多大就不用 malloc+read 多大)、按需读盘(page fault 时才 I/O)、多 worker 共享物理页。

**MAP_SHARED vs MAP_PRIVATE:** `PROT_READ` 下两者读行为完全一样(都不能写、都不污染文件),但 `MAP_SHARED` 让**多个进程映射同一文件时共享同一份物理页**——serving 里多 worker 加同一模型能省内存。这是 llama.cpp `llama_mmap` 的实测选择(源码级核对,非记忆)。

### 2.5 零拷贝怎么证明(别名法,不碰裸指针)

不靠"比两个指针相等"——那只能证明地址碰巧一样。改用别名法:造视图读一次 `out0` → 原地改底层文件(同大小 `r+b` seek+write)→ 同一视图再读 `out1` → 若 `out1` 反映新文件内容且 ≠ `out0` 即零拷贝;若等于旧快照即被偷偷拷贝。

**为什么这比指针比较强:** 它证明的是"视图实时读 mmap 活页",正是 serving 零拷贝的真实价值。且不暴露裸指针给 Python(pybind11 把 `void*` 默认绑成 `PyCapsule`,Python 永不碰地址),符合工业原则。

### 2.6 防御性校验与设计诚实性

- **offset 对齐校验:** bf16 要 2 字节、float32 要 4 字节,`offset % dtype_size != 0` 抛异常,否则访问错位地址 SIGBUS/UB;
- **越界校验防整数溢出:** `(offset > size) || (nbytes > size - offset)`。**先判 `offset > size`** 保证 `size - offset` 不下溢(无符号减下溢成超大值),用减法避免 `offset + nbytes` 加法溢出绕回小值漏判。这是防整数溢出的经典模式;
- **`device` 参数该不该有:** `make_weight_view` 故意不收 `device`——mmap 产物天生是 CPU 地址,收了就是误导 API(调用方以为能选 GPU,实际深处抛"CUDA not implemented")。职责分离:mmap→CPU 视图(本函数),CPU→GPU 是另一个操作。**一个"只能取一个值"的参数不该存在。**

### 2.7 分层与 Python 互操作

**三层架构,依赖单向:**
- core 静态库(`xstar_cpp/{include,src}`):纯 C++ 逻辑(Tensor/bfloat16/MMapFile/UniqueFd),不碰 Python;
- weight_io 组合层:造权重视图,依赖两个底层头、底层头互不依赖;
- 绑定层(`bindings/`):只暴露已有符号,不写业务逻辑。

**互操作要点:** 静态库链 `.so` 必须 `POSITION_INDEPENDENT_CODE ON`(否则 "recompile with -fPIC");`pybind11 + std::vector` 必须 `#include <pybind11/stl.h>`(漏了下游诡异地堆损坏);pybind11 3.x 的 `py::arg` 数必须严格等于函数 arity(否则编译期静态断言失败)。

## 三、踩过的坑(每个都是真 bug,非理论)

1. **初始化列表 move 陷阱:** `shape_(std::move(shape))` 后,body 里**绝不能再碰被 move 的参数 `shape`**,要用成员 `shape_`。否则 `shape.size()` 返回 0 → strides 空 → numel 走成员(对)与走局部(错)不一致 → memcpy 写越界 → 堆损坏。ASan 定位;
2. **`std::move` 参数后的使用:** move 完参数处于"有效但未指定"状态,函数体内只能用成员;
3. **析构抛异常:** 栈展开时第二个异常 → terminate。所有析构 `noexcept` 或吞掉错误;
4. **move 赋值自赋值缺 return:** `if (this != &other) {...}` 结构,自赋值分支掉到函数末尾没 return → 非空函数 fall-off = UB。`-Werror=return-type` 编译期抓 + UBSan 运行期抓。修法:开头 `if (this == &other) return *this;` 早返回;
5. **move 赋值必须判自赋值 + 先释放既有资源:** 不判自赋值 → 先 release 自己(关掉 other 的 fd)再偷 → 悬空;不先释放旧资源 → 旧 fd/buffer 泄漏;
6. **悬挂引用:** 返回 `Tensor&` 但返回的是局部对象 → 引用指向已析构对象 = UB。move-only 类型应**按值返回 + RVO/move**(`make_unique`/`make_shared` 惯例),非返回引用;
7. **`explicit` / 默认参数只在声明(.h)出现一次:** 定义(.cpp)重复写 → "redefinition of default argument";
8. **`void*` 不能做指针算术:** 按字节移动要转 `const char*`(或 C++17 `std::byte*`),`+ offset` 才是字节单位;
9. **头文件自洽:** `weight_io.h` 自己用 `int64_t` 就自己 `#include <cstdint>`,别依赖传递包含——传递链断了就崩;
10. **`aligned_alloc` 要求 size 是 alignment 整数倍**,size=0 行为未定义——要 `(nb+63)&~63` 对齐 + 0 给最小 64;
11. **`-fsyntax-only` 不可靠于流敏感问题:** return-type 这类控制流问题要真实 `-c` 才抓得到,fsyntax-only 会漏。

## 四、能调用的工具肌肉记忆

- **ASan:** `-fsanitize=address` + `LD_PRELOAD=$(gcc -print-file-name=libasan.so)`——定位 C++ 堆损坏比看 "malloc unsorted" 瞎猜强百倍;
- **UBSan:** `-fsanitize=undefined -fno-sanitize-recover=all`——抓 fall-off UB、整数溢出、use-after-move;
- **`-Werror=return-type`:** 把"控制流掉出非空函数"从警告升成编译错误;
- **别名法证明零拷贝:** 不碰裸指针,改底层文件看视图是否实时反映——可复用到 Phase 2 KV cache 共享验证;
- **探针 + 多工具交叉验证:** 一个怀疑点用 `-Wreturn-type` / `-Werror=return-type` / UBSan 三个独立工具坐实,不靠单一信号下结论。

## 五、能在后续阶段带走的方法论

1. **RAII 包一切资源:** fd、映射、buffer 都用守卫/类包,析构兜底,业务路径才手动检验。手写清理每多一个分支就多一个泄漏点——守卫从结构上消灭它;
2. **析构不抛是铁律:** 写 RAII 类第一件事想"析构失败怎么办",答案是 log + swallow,不是 throw;
3. **move 语义要传播所有所有权标志:** 不只搬指针,`owns_data_` 这类标志必须一起搬,否则两条析构路径会串台;
4. **按语义命名,不按实现命名:** `make_weight_view` 而非 `tensor_view_from_mmap`——M5 换 safetensors 实现时名字不用改;
5. **设计诚实性:** 一个参数若只能取一个有效值,就不该存在;收了就是误导调用方;
6. **整数溢出用减法避加法:** 边界校验 `a + b > limit` 改 `(a > limit) || (b > limit - a)`,先判 `a > limit` 防减法下溢;
7. **测试跟着 milestone 走,不攒到最后:** op + 测试一起提交,刚写完最清楚"该输出什么";对拍即测试,judge 通过 = 测试通过;
8. **基础设施测试 vs 对拍测试分文件:** `test_cpp_tensor.py`(纯断言,不要 oracle)vs `test_cpp_ops.py`(挂 oracle judge 对拍);C++ 侧 `xstar_cpp/tests/` 留给纯单元 + 内存安全(不经 Python,上 ASan/UBSan 友好);
9. **验收要持久化:** 一次性验证脚本放 /tmp 重启就丢;沉淀成 pytest 才是可回归的事实;
10. **讲工业做法只讲确定的:** 不确定先查源码/文档,不臆断(本轮 MAP_SHARED、pybind11 caster 都查了源码坐实,纠正了多处凭记忆的倾向)。

## 六、M1 交付状态

- ✅ C++ Tensor 运行时(dtype/device/bfloat16 RNE/tensor owned + 非拥有视图);
- ✅ pybind11 绑定(from_numpy/to_numpy/Tensor/MMapFile/make_weight_view),`void*` 绑 PyCapsule 不暴露裸指针;
- ✅ CMake 分层(core 静态库 PIC + pybind11 MODULE + .so 直出包目录 + export compile_commands);
- ✅ MMapFile + UniqueFd RAII(MAP_SHARED + PROT_READ,对齐 llama.cpp);
- ✅ weight_io.make_weight_view 零拷贝(对齐校验 + 越界减法防溢出 + 不收 device);
- ✅ 内存安全:ASan 无 double-free/leak,UBSan 无 fall-off UB;
- ✅ 验收持久化:`tests/test_cpp_tensor.py` 10 项全绿,可一键回归。

**M1 闭环。地基就位,下一步 M2:rmsnorm 接 oracle judge 对拍(Phase 1 算子主体开始)。**