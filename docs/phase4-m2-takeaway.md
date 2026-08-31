# Phase 4 M2 能带走的东西：RadixAttention 接调度器(准入 pin 驱逐 + 抢占 glue)

> 格式对齐 `phase4-m1-takeaway.md`："可带走的东西"每条 = 一个带具体坑的认知 + 失效场景 + 为什么。
> M2 = Phase4 第二块：把 M1 的 RadixTree 接进 continuous-batching 调度器 —— 准入 match→pin→evict→fork→adopt、finish insert→dec_lock_ref、decode 抢占。无新 kernel，全是 scheduler.py 的 glue 层；外带一个靠 gdb 才定位到的 UAF 段错误。

---

**1. 准入五个动作的顺序是决策不是顺手：pin 必须在 evict 前，否则驱逐先踢掉刚匹配到的共享前缀**

- scheduler.py `_tick` 的顺序：gate(`budget+evictable-reserve-own < needed_new` 就 break，L151)→ pin(`inc_lock_ref(node)`，L160)→ evict 补缺口(L164)→ fork+adopt(`adopt_prefix(bm.fork(blocks))`，L186)→ residual prefill。
- 为什么 pin 在 evict 前：evict 弹的是 LRU 里可驱逐叶子，match 到的前缀若没 pin 住，12×256 准入的驱逐风暴(`needed_new-budget` 很大)可能把它自己弹走 → adopt 拿到已释放 block。
- 但 pin 不是无脑乐观：evict 不够仍有 `dec_lock_ref(node)` 回滚(L170)——"先锁再验驱逐"的两阶段。承接 M1 第 3 点：lock_ref 管"前缀能不能被踢"，ref_cnt 管"物理块能不能回收"，pin 只碰前者。

**2. `match_prefix(seq[:-1])` 的 `-1` 是防"全匹配空查询"——"树只认块对齐前缀"在调度器侧的落地**

- L142 故意切掉最后一个 token(残余 < block_size 不参与 match)，注释写死"空残差截断，防止全部匹配共享，没有 query"。
- 失效场景：match 整条且恰整块对齐 → `needed_new=0` → 无 query 空转。
- 承接 M1 第 4 条"选 B residual 不进树"：`-1` 是"caller 持残余、树管块对齐前缀"在 scheduler 侧的对称实现。

**3. decode 预留 = `num_free() >= len(running)` 已经是"每请求预留 1 块"——我提议加 precheck 是错的，被纠正**

- decode 只在 block 边界 alloc 一次 1 块(paged_kv_cache.cpp `cursor_ % block_size_ == 1 → alloc(1)`，L76/154)。所以 `while num_free() < len(running): preempt`(L199)等价于"给每个 running 留 1 块才出循环"，已覆盖 decode 上限。
- 我第一版提"decode 分支加容量预检查"，理由是"decode 也会 alloc 没保护"——错。`== len(running)` 就是那 1 块的预留。
- 教训：判别据要看隐含边界条件，别看到"无显式 precheck"就默认无保护。跟 M1 第 8/9 条"脑内枚举不可靠"同源。

**4. 本次核心 bug：insert 挂孩子不查 `now->in_lru` → 违反"LRU 只挂无子未 pin 叶子"不变量 → 孤儿悬空 parent → UAF**

- M1 三处 push 都守 `children.empty()`(dec_lock_ref L231、evict 级联 L247)，即 LRU 里只有叶子；但 `insert`"没这叉加"分支(L150)在已有叶子 `now` 下挂 child 时**没查 `now->in_lru`** → 带孩子的节点留在 LRU。
- 崩链：evict 弹 `now` 并 delete → child 变孤儿、parent 悬空 → 下次 evict 弹孤儿、`_delete_leaf`(L302)解引用已释放 parent → SIGSEGV。
- 触发形：16×80 留下的 gen(80) 叶子被 12×256 序列严格延长 → insert 停在该叶子下挂孩子；pool=150 驱逐风暴很快把孤儿弹出来。flaky 因为"挂孩子 + 随后 evict"两时序条件要同框。
- 修法三行(在 `lru_.push_back(leaf)` 前)：`if (now->in_lru) { evictable_blocks_ -= now->block_table.size(); lru_.remove(now); }`(现 L202-206)。回归 `test_insert_lru_invariant`。
- 教训：结构不变量要在**每个**"改 parent/孩子关系"的转移点都查，不能只守 push 点。M1 守了三处 push，漏了 insert 这个"加孩子"点——不变量是全局的。

**5. 三种"进程死了"要分层鉴别：Python 异常 / 后台 task 静默崩 / 信号级崩溃——这次是第三种**

- M3 第 4 条是第二种(binding 漏一处 → AttributeError 在后台 task 抛 → 静默 hang)。M2 这次是第三种：UAF 是 SIGSEGV，不是 Python 异常，`_loop` 的 `except Exception` 接不住，`_loop crashed` 一次没打。
- 判别：`_loop` 包 try 后 `_loop crashed` 计数 = 0 但 process 已死(Connection refused)→ 不是异常路径。
- 我一度往"asyncio 强引用 task 静默吞异常"想是错的——那条路会打印 "Task exception was never retrieved" 且只对弱引用场景。实测终端零输出就排除。
- 三层次各有工具：异常 → `_loop` try；后台 task → loop exception handler；信号级 → faulthandler(Python 帧)+ gdb(原生帧)。

**6. gdb 定位法链：faulthandler 锁 Python 帧 → gdb 锁原生帧 → 回源码找结构不变量**

- faulthandler 崩溃瞬间打出 `scheduler.py:164 in _tick`(= `radix_tree.evict(...)` 调用行)——只到 Python 帧，看不到 C++ 崩哪行。
- `cuda-gdb -batch -ex run -ex bt` 抓到原生栈：`_delete_leaf → map::erase → operator< → vector<int>::end()`——`std::map<vector<int>,RadixNode*>` 走到这步直接指向"children 在操作垃圾 key"，回源码就是 parent 悬空。
- memcheck 太慢走不到崩溃轮，gdb 对。承接 M8/M6"反向探针"：一层工具测不到的换更深一层，这次是工具降层 from Python 到 C++。

**7. evictable_blocks 按段存：node-level LRU 有孩子不 evictable，计数是"可驱逐节点各 block_table.size() 之和"不是"树里所有块"**

- 回归里 insert 延长序列后 `evictable_blocks()==2` 我一度以为错(以为该 4)。实际：新 leaf 只存新增段 `block_table[matched/bs : total/bs]`(insert L199-200)，父有孩子后从 LRU 摘出不可驱逐，evictable = 只有新叶 2 块。
- `freed=4` 是先弹 B(2)→ A 变无子被级联推回 LRU(evict L247)→ 再弹 A(2)。
- 教训：验计数断言前先弄清语义口径，别拿直觉值反推实现。跟 M1 第 10 点"evict 拆成两个关切"同源。

---

## 共性

M2 教两件事：**glue 层顺序即契约**(五动作相对位置都是决策，换序踩共享前缀被踢/空查询/decode 溢出)和**结构不变量全局自洽**(insert 漏一个检查，数据结构"病历"就潜伏成 flaky 段错误)。debug 这条最值：三种"进程死"要分层鉴别，单看代码全误判——M2 我连续误判"decode 无保护""asyncio 静默吞""empty decode batch"三个方向，靠 faulthandler + gdb 实锤才收敛到 `_delete_leaf`；跟 M1 结尾"连自己给的建议都要过跑出来看"同一条，只是 M2 的"跑出来"是 gdb 不是 pytest。
