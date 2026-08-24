# Phase 1 能带走的东西:解析器 / 整模型 / mmap 全量 / 端到端

> 格式对齐 M6 softmax takeaway:每条 = 一个带具体坑的认知 + 失效场景 + 为什么。
> 本文覆盖 M9 收尾的四块(parser / 整模型 / mmap 全量 / 端到端),与 [phase1-m1-takeaway.md](phase1-m1-takeaway.md)(M1 RAII / mmap 深挖)互补。

## 解析器 (json_scan + config + safetensors header)

- **手写递归下降,不引 nlohmann**:一是课程自包含(不指第三方实现);二是 config 要同时做两件相反的事——**必填字段缺失抛带名字的错**(默默用 0/garbage 会让少了 `num_key_value_heads` 的 config 跑出静默错误结果,debug 到死)、**未知字段要跳过**(HF 每版 config 加新字段,严格 parser 每版都崩)。「严格 vs 容忍要分对象」:必填字段对未知值严格、对新增字段容忍。
- **`read_number` 的 bounded scan**:`strtod` 会「能解析多远就解析多远」(parse as far as it can into the buffer),无法自己停在下一个 `,`/`}`。所以必须**先手扫数字合法字符(`0-9` `.` `e/E` `+` `-`)给字面量定界,再把这段 null-terminated 子串喂给 strtod**,offset 用 `end_ptr - p`(已扫长度)不是裸指针比较。跳过 strtod 直接手算 = 重新实现 IEEE-754 转换,错路。
- **`read_uint` 显式拒 `-`**:非负整数解析,不拦 `-` 的话 `val = val*10 + (c-'0')` 会把 `-1` 吃成垃圾。拒 `-` 比「之后查 val<0」早、报错点准(直接指到符号位)。
- **`skip_object`/`skip_array` brace-matching 跳未知对象**:扫到 `{`/`[` 深度计数配对跳过,不解析内部——这是「未知字段容忍」的物理实现,且对 `__metadata__` 这种任意嵌套对象通用。
- **safetensors header = 8 字节 LE 长度前缀 + JSON**:前 8 字节 uint64 little-endian = header 字节数,之后是 JSON(每 tensor 的 dtype/shape/data_offsets)。两段:先读长度、切 header、再 parse。`__metadata__` 用 skip_object 跳过(只关心 tensor 描述符)。
- **每处 `*p` dereference 前必有 `p == end` 守卫**:bounded scan 的命门——「先读再判越界」会在 malformed header 上越界读。写成 `if (p == end) throw` 在前、`*p` 在后,且 throw 带位置信息。
- **绿的保护伞**:config 只用真模型(字段全对)→ 造「必填字段缺失」「未知字段混入」;header 只用干净 safetensors → 造「truncated header」「未知 dtype 字符串」「data_offsets 跨度 ≠ nbytes」。trailing comma 容忍要显式测(严格 JSON 拒、lenient 接受)。

## 整模型 (qwen2_model.cpp 编排 + loader)

- **只有 `gate_up` 是 owned,其余全是 mmap view**:loader 里所有权要想清——`gate_up` 是 C++ 侧要拼的 concat(safetensors 存 gate/up 两个 key,block op 吃融合的),**唯一一次新分配**;其余(embed / per-layer / ln_final)全是 non-owning view 指向 mmap,含 `lm_head` 是 `embed_tokens` 的**第二个 view**(weight tying)。owned 尽量少、view 尽量多 = 零拷贝、零重复存储。
- **`get_weight` 用 `find` 不用 `operator[]`**:meta map 是 `const` 引用传入,`operator[]` 无 const 重载(只在 key 不存在时插入,破坏 const)——编译错。且 `operator[]` 是「找不到就插入」,对「必填 key 缺失要抛错」**语义相反**(你要找不到→报错,不是找不到→造默认值)。正解:`find` 拿迭代器、判 `== end` 抛带 key 名的错、用迭代器的 value。一次查找,不是 find + 3×operator[] 的 4 次。
- **`gate_up` cat 顺序 gate 前 up 后,load-bearing**:SwiGLU 是 `silu(gate)*up`,gate 过 silu、up 不过。cat 顺序反了 → silu 吃到 up、up 直接乘 gate,静默错(数值不像 NaN 那么明显)。顺序由 loader 钉死,注释写明「reversed order silently corrupts SwiGLU」。
- **lm_head tied 是 view 不是拷贝**:`lm_head_w` 指向 `embed_tokens_w` 同一 mmap region,都 non-owning、都不 free、无 double-free。HF `tie_word_embeddings=true` 对应到这里就是「不存独立 lm_head.weight,290 keys = 24×12 + embed + norm,没有 lm_head」。tied=false 是另一条路(要存独立权重),本模型用不到,loader 直接拒并报明。
- **forward 是 black box,无中间态 port**:整模型 forward 只返回 logits,不暴露「返回第 k 层 hidden」的 debug hook——推理主路径保持干净。端到端 parity 挂了靠**链式定位**:embedding bit-exact 抽查 → 单 op 回归 → layer-0 block 抽查 →(必要时)ad-hoc 前 k 层探针。不为了 debug 在生产 forward 上开口子。
- **`qwen2_forward_py` 的 positions=None → arange materialize (path B)**:core 的 rope 无条件解引用 `positions[i]`,nullptr 必崩。决定是 binding 在 None 时造 `arange(seq_len)`——core 永远拿非空,rope 保持「positions 必非空」的无状态设计。arange 不是 RoPE 的固有语义(对比 attention 的 mask=None 在 op 内建 causal——causal **是** attention 的固有默认),所以 arange 不进 rope op、放 binding。`std::optional<vector>` 的生命期要盖过 `qwen2_forward` 调用(函数体 scope,不是 else-block scope——else-block 结束 vector 析构、指针悬空,rope 的 range check 会把悬空读成「out-of-range position」而非 segfault,反而是好事)。
- **绿的保护伞**:loader 只用真模型 → 造「缺一个 key」「tie=false」「config 和 weights 来自不同模型(num_hidden_layers 不匹配)」;forward 只测端到端 → 加 layer-0 block 抽查隔离「是某层坏还是编排坏」。

## mmap 全量 (MMapFile + weight_io + bridge)

- **权重从不进 RAM**:`mmap(2)` 映射整个 safetensors,header 是 length-prefixed JSON(前 8 字节 uint64 LE),parse 出每 tensor 的 offset/shape/dtype,C++ `Tensor` 直接是**指向映射页的 non-owning view**。5GB 模型 0 内存拷贝、启动即用、按需 page-in。`make_weight_view` 算 `addr + 8 + header_len + data_start_off + *data_start_off` 拿 tensor 起点。
- **RAII + fd 不存成员**:`MMapFile` 析构 `munmap`;**fd 在 mmap 成功后立刻 close**(mapping 自带文件引用,POSIX 下 mapping 比 fd 活得久,munmap 不需要 fd)。不存 fd 成员 = 避免一个可能被 double-close 的 stale fd。构造失败要手算补偿:构造体抛异常**不触发析构**(只析构成员/基类,标量无析构),所以 mmap 失败后的 `munmap` 要在 throw 前手动做,否则泄漏。
- **view ctor 不查对齐,真守卫在 `make_weight_view`**:`ptr = mmap_base + offset` 的对齐取决于 **offset 不是 mmap base**(mmap base 4KiB 页对齐,`base+offset` 未必对 dtype 对齐)。view ctor 信任 caller;`make_weight_view` 查 `offset % dtype_size == 0`。注释别写「mmap 4KiB trivially OK」——那是错的安慰,对齐看 offset。
- **offset 越界用安全减法防溢出**:bounds check `view_end <= mmap_end` 不能直接 `start + size <= end`——`start + size` 可能 int64 溢出绕回小数,假通过。先 `if (size > end - start) throw` 再用差,避免加法溢出。「check 在它保护的运算之前」的一员。
- **bf16 跨语言 bridge 走 uint16 重解释,不是 f16**:bf16 ≠ f16(不同格式)。唯一 bit-exact 的跨界是 `t.view(torch.uint16).numpy()`(16 个 bf16 bit 当 uint16)→ C++ `from_numpy_raw(..., BFloat16)` 重解释回去;反向 `to_numpy_raw .view(uint16) .view(torch.bfloat16)`。**这条 round-trip 是所有 parity 判断的命门**——它位级错了,上层所有 allclose 都在测 bridge 误差不是 C++ 正确性。f32 直接拷。
- **`cpp_to_torch` 的 `ref_shape` 是信任陷阱**:xstar Tensor 出来是 flat uint8 buffer(`to_numpy_raw`),没形状,`ref_shape` 外部传。reshape 到 `ref_shape` 若 C++ 真实 shape 元素数同但维度不同,**静默成功、掩盖 shape bug**。正解:调用前 `assert cpp.shape() == ref.shape`(test_cpp_qwen2_model.py 有这个 pattern),bridge 不替你兜。
- **绿的保护伞**:bridge 只测「形状匹配」→ 造「元素数同但维度不同」(应被 assert 挡);mmap 只测小文件 → 造「空文件」「header 长度 > 文件大小」「data_offsets 跨度 ≠ nbytes」。

## 端到端 (parity harness + greedy 验证)

- **两层验收:单层强 judge + 端到端弱验收**:单层 per-layer allclose(`TOLERANCES` 表:embedding equal / rmsnorm 1e-6 / rope 1e-2 / attention 2e-2 / mlp 5e-3 / block 0.2 放大)——证明每个 op port 忠实;端到端走 **greedy 20 步 + 流畅生成 + argmax 一致**,不是 allclose。
- **端到端不能 allclose,为什么**:24 层 bf16 累加顺序差异让 logits 偏 ~0.578125,但 argmax 稳、文本流畅。强行 allclose 会把「bf16 跨 24 层的累加序漂移」(无害)判成「port 错」(假红)。**强 allclose 留给单层,端到端只看生成质量**——对拍分层强弱点设计。
- **`max_diff` 恒定不随 seq_len 涨 = 误差来自单步 matmul 不是层间累积**:max_diff 在 20 步里**恒定**(seq_len 从 5 涨到 25,diff 不动)→ 误差是每步 lm_head matmul 的 bf16 累加序差异(每步重算),不是 24 层 hidden 的累积。一个验收 harness 不只能验货,还能**反向定位误差源**。
- **参考必须跑 CPU**:C++ 是 cpu-only。若 ref 跑 cuda,你把「CUDA bf16 vs CPU bf16」和「C++ port vs xstar-Py port」两个误差源混在一起,污染判断。**实验里只有一个变量**的纪律搬到工程对拍——device 对齐是底线。
- **model/lm 不进 judge 表**:judge 的 `TOLERANCES.get(layer)` 对 model/lm 返回 `(None,None)` → 抛 ValueError。这是设计不是疏漏:整模型/端到端走 coherence(流畅 + argmax),不走 allclose。强 judge + 弱 end-to-end,各管各的。
- **NaN 根因是未初始化 bias,不是 einsum 不是 RMSNorm**:`Linear` 默认 `bias=True` → `torch.empty` 未初始化 bias;`Qwen2ForCausalLM` 构造 lm_head 没传 `bias=False`(HF 的 lm_head 无 bias);loader 不复制这个不存在的 key → 残留内存含 nan/inf → 加到干净 matmul 上 → NaN logits。**触发是非确定的**(`torch.empty` 残留跨 run 变),所以 parity_qwen2(cuda)曾「碰巧过」,修了之后「必然过」——输出不变是 expected。教训:未初始化内存 bug 是非确定的,「它过了」不等于「它对」。
- **绿的保护伞**:端到端只测「生成流畅」→ 注入 lm_head 未初始化 bias(应让流畅度崩 / argmax 全错);单层只测「allclose 过」→ 注入 ln2 喂错张量(应让 block allclose 红,证明容差 load-bearing,见 M8 ablation)。

## 共性

Phase 1 教的不是「跑通 transformer」,是「让每一步可验证、可解释、可几个月后唤回」——parser 向前兼容、loader 零拷贝 + view 所有权、bridge 位级精确、对拍分层强弱 + 单变量 + 反向定位。
