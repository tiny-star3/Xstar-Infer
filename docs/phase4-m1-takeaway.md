# Phase 4 M1 能带走的东西:RadixCache 数据结构

> 格式对齐 `phase3-m3-takeaway.md` 的"可带走的东西":每条 = 一个带具体坑的认知 + 失效场景 + 为什么。
> M1 = Phase4 第一块:RadixNode + LRUList + match/insert/split + inc/dec_lock_ref + evict + 级联。未接调度器,所以没有 bench 数,只有结构决策 + 实现坑。

---

**1. LRU 放哪层是 Phase4 的第一个结构决策:node 层(SGLang 式),不是 block 层(vLLM 式)**

- 路 X = node-level LRU(node 挂 lru_prev/next,驱逐单位 = 整段 node),路 Y = block-level dual-tail(free-list 加 tail,block_hash 命中走 tail)。选 X 的依据是核实 SGLang `radix_cache.py`:`evict` 从 `evictable_leaves` pop 叶子 node、`free_segment(x.value)` 释放整段;`inc/dec_lock_ref` 沿父链走,lock_ref 在 node 层。**BlockManager 不动**(保持 M7 单一 LIFO free-list),LRU 上移 RadixTree 层。
- 废弃路 Y 的原因:block_hash 是 vLLM 的配套方案,radix 下 block 层无法知道一个 block 属于哪个前缀、该不该走 LRU——layering 乱。
- 承接:这是"计划随决策更新"——M7 header 注释的"加 tail"假设基于"Phase4 用 block_hash",现在定 radix,那个假设作废,不是 M7 写错了。

**2. heap vs 双向链表:heap 是 SGLang 支持多策略的代价,只做纯 LRU 的我们不付**

- 核实 SGLang `evict_policy.py`:全部 7 个策略(LRU/LFU/FIFO/MRU/FILO/Priority/SLRU)用同一 heap 机制,差别只在 `get_priority(node)` 返回的 key。heap 是"一套机制带多策略"的统一抽象。
- 我们只做 LRU → 双向链表(命中 `move_to_back` O(1)、驱逐 `pop_front` O(1)),省掉 heap O(log n) 的重插。写法复用 M7 BlockManager 哨兵双向链表——同一结构、不同层级。
- **失效场景(第 4 条的核心)**:match 命中不刷尾 → LRU 退化成 FIFO,热前缀被误踢。双向链表的 O(1) move_to_back 是"LRU ≠ FIFO"唯一差别的基础设施。

**3. 两层计数别混:node.lock_ref 驱驱逐,block.ref_cnt 驱物理释放**

- lock_ref 在 node 层、inc/dec 沿父链向上,管"这个前缀能不能被 evict";ref_cnt 在 bm 层、fork +1 / free −1,管"这个物理块能不能回收"。evict 唯一释放点是 `_delete_leaf → bm.free(node->block_table)`,别处不释放。
- 失效场景:把驱逐和 free 混成一个动作 → 共享前缀的 block 被释放,另一个还 pin 着它的请求读到脏块/double-free。
- 承接 M3 第 2 点"reset 释放自己拥有的、不释放不拥有的"——所有权契约拆成两个计数,每个只干一件事。

**4. 选 B(residual 不进树):我最初推荐选 A,核实源码后纠正**

- 我第一版推荐"残余段也进树"(选 A),理由是"树里信息更全"。核实 SGLang 后发现是选 B:`page_aligned` 截断(L150)、`cache_unfinished_req` 里 L565 注释写死 partial part 不加进树、`cache_protected_len` 字段专门用来在下一轮把它释放掉。
- 为什么工业这么选:residual < block_size 进树会破坏"驱逐单位 = 整块",split 会出现半共享块,树形和物理块不再一一对应。caller(请求级)持有残余,树只管块对齐前缀。
- **教训:推荐设计前先查工业实现,不是讲到不确定才查**。这条是 feedback-research-industrial-before-design 的又一次兑现,这次是被自己抓的。

**5. `insert` 完全命中返回现有 node、不覆盖 block_table —— 防"重复 insert 已缓存前缀"**

- CONTRACT(docstring 写死):insert 记录的是 caller 已 alloc + prefill 过的 block_table,树不分配;完全命中时返回已有 node,**不覆盖**。
- 失效场景:两个请求共享前缀,第二个再 insert 同一段 → 覆盖会把第一个请求 pin 着的共享块号换成第二份,前缀复用读到错误物理块。
- 测试兜底:`test_insert_full_match_no_overwrite` 断言第二次 insert 返回的 node.block_table 仍是 `[10]`、lru_size 不增。

**6. pybind11 裸指针返回值默认 take_ownership → double free;修一个要扫同类**

- `match_prefix` 返回 `std::pair<int, RadixNode*>`,pybind11 对裸指针默认 take_ownership → Python 侧丢弃时 free 一次,tree 析构再 free 一次 → `free(): double free detected in tcache 2`。**崩点在 `del t` 之后**(`clean ok` 打印完才崩),定位到 tree 析构阶段,不是 match 调用点。
- 修法:绑定层 lambda 手动 `py::cast(node, py::return_value_policy::reference)`,C++ 零改。`insert` 返回 leaf 裸指针,**同 bug**——match 修了复测才发现 insert 也在崩,第二次才补上。
- 承接:M6 loader "tied lm_head 非拥有 view"、M3 第 4 点"binding 三处同改漏一处"——**同一类:边界处的所有权/契约要显式,且修复要扫同类调用点,不能只修崩的那一个**。

**7. build 警告三连:递归 lambda 自引用 + 两处 -Wreturn-type**

- 递归 lambda `auto DeleteTree = [](auto&& self, RadixNode* now){ self(self, now); }` → GCC "use of lambda before deduction of auto"(**lambda 还没推导出自己的类型就要用它**)→ 显式 `std::function<void(RadixNode*)>` 解决。
- `pop_front` 算完 result 没 `return result` —— 同 phase2-m1 第 6 点,非 void 没 return 是 UB;`insert` 尾部加不可达但编译器不知道的兜底 `return now`。
- 承接 phase2-m1 第 6 点"build 警告比跑测试快"——先 build 干净再跑测试。

**8. test_split_throws 的构造陷阱:异常分支先确认入口可达,不可达的防御分支不值得测**

- 我最初给的建议:"分叉点在 block 中间 → throw",输入 `[1,2,3,9,...]`。跑了 DID NOT RAISE。
- 根因:**root 的 key 为空**,`first_block = tokens[0:4] = [1,2,3,9]` 在 root.children 就找不到 → 走"没这叉,新建 leaf"分支,根本不进 `_split_node`。
- 深层:tree 只在 block 粒度分叉,children 的 key 就是整块 first_block,不匹配就成新叉,**insert 正常路径永远产生不了非块对齐的 split_len**。`_split_node` 的 throw 是内部不变量防线(防御性),不是用户可触发的异常路径。
- **教训:造负向/异常测试 case 之前,先确认这条路径从真实入口是否可达**。不可达的防御分支不值得测,测了还给出"这段逻辑被覆盖了"的假信号。
- 这个 case 我自己也滑了一跤(先推荐了错误构造),**脑内枚举场景不可靠,跑出来看**——M1(Phase3)"probe 分层验证"在测试设计层的复用。

**9. split 后的 LRU 归属:旧 child 仍在 LRU、上层共享 node 不进,靠 evict 级联兜底**

- 实测 split 后 `lru_size == 2` = 旧 child(缩短,in_lru 未动)+ 新 leaf;上层共享 node lock_ref=0 但非 leaf,不进 LRU。
- 我一度担心"上层 node 永久孤儿、不可驱逐"——不成立:`evict` 的级联(delete leaf 后 parent `children.empty() && lock_ref==0 && !in_lru` → push_back)保证两个分支都被踢光后 parent 才入 LRU 被踢。**有 sibling 就不级联**是设计:驱逐它会让两个分支同时失效。
- 又一次臆断被打脸:担心之前先跑 + 读级联代码,不是脑内推演。

**10. 测试分四批,依赖逐层解锁:P0 纯树 → P1 split → P2 lock_ref → P3 evict(需真实 bm block id)**

- evict 路径 `bm.free(block_table)` 要求 block_table 来自**同一个 bm 的真实 alloc**——树内自造 id(如 `[10]`)会抛 "block not allocated"。P3 的 case 用 `bm.alloc(2)` / `bm.alloc(1)` 再手拼 `[block_table[0], block_table2[0]]`(共享前缀 block id 必须一致,真实流程由 match→fork→insert 闭环保证,测试里靠手工拼,注释写明)。
- **block_size 口径**:测试用 4 方便手算,接调度器时必须 == bm.block_size()=16(契约写死)。测试值和接入值不是一个东西。
- 承接 M3 教训"分层验证,一个 case 只测一件事":evict 拆成"树踢不踢"(lru_size 变化)和"free 对不对"(bm 耦合)两个关切,不混在一个 test 里。

## 共性

Phase4 M1 教的是**"结构决策先核工业实现,实现坑靠 build + 绑定边界 + 分层测试暴露"**:路 X/选 B/不上 heap 三个决策都是核实 SGLang 源码定的(不是凭印象);double free 和 return-type 是 build + 崩溃栈定位的;split_throws 和"孤儿"两个误判都是**跑了才纠正**——脑内枚举在数据结构边界上不可靠,这个阶段连我自己给的建议都要过"跑出来看"这道门。
