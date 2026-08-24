# Xstar-Infer 分布式多语言推理引擎

**项目定位：** 针对 DeepSeek-V3 (MoE + MLA) 架构深度优化的工业级高并发大模型推理引擎  

摒弃学术界动态图，纯前向高性能设计；融合 C++ 底层显存池与 Python 异步连续批处理；通过 Raft 共识实现大规模 MoE 专家并行的高可用控制  

## 系统架构与语言分工 (The Polyglot Architecture)

系统采用微服务与多语言混合架构，严格坚守 “让合适的语言做合适的事” 的工业界最佳实践  

|   **模块层级**   |           **技术栈**           |                    **核心职责与输入输出**                    |
| :--------------: | :----------------------------: | :----------------------------------------------------------: |
| **接入与调度层** | **Python (FastAPI + asyncio)** | 负责 HTTP/gRPC 协议解析、Tokenizer 分词，以及基于事件循环的 Continuous Batching 组装 |
|  **语言绑定层**  |       **C++ (PyBind11)**       | 打通 Python 调度队列与 C++ 计算引擎，实现张量指针与元数据的零拷贝传递 |
| **张量与内存层** |    **C++ (重构版 Needle)**     | 剥离计算图追踪，纯前向张量库；基于 Radix Tree 的 PagedAttention 物理显存块管理 |
| **高性能算子层** |     **CUDA C++ / Triton**      |  手写 Fused MLA Decode 算子、并行 Top-K 采样，压榨 HBM 带宽  |
| **分布式控制面** |      **Go (gRPC + Raft)**      | 独立微服务进程。监控 C++ 推理节点的健康状态，维护 MoE 路由拓扑拓扑图，实现故障剔除 |

## 核心模块功能需求详细拆解 (Requirements)

### 纯前向 C++ 张量运行时 (Tensor Runtime)

- **彻底剥离 Autodiff：** 移除原有框架中所有的 `requires_grad`、`backward()` 逻辑及计算图数据结构（Tape），消除 Save-for-Backward 导致的显存暴涨  
- **In-place 算子签名：** 所有底层 CUDA 算子修改为原地操作或目标地址注入（如 `void matmul(Tensor& A, Tensor& B, Tensor* Out)`），消除动态内存申请 (`cudaMalloc`) 带来的 CPU 阻塞  
- **Safetensors 零拷贝加载：** 实现基于 `mmap` 的文件解析器，绕过 Python，直接将磁盘上的模型权重（如 FP16/FP8 数据）通过 `cudaMemcpyAsync` 映射并载入 GPU  

### PagedAttention 与 Radix 前缀缓存系统

- **物理显存池 (Block Pool)：** 引擎启动时接管 GPU 80% 的空闲显存，将其等分为固定大小的 Block（如每个 Block 存储 16 个 Token 的 MLA 隐变量 c_t）  
- **Radix Tree 缓存管理：** 引入基数树管理多轮对话的上下文。树节点映射到具体的物理 Block，支持跨请求的系统提示词（System Prompt）缓存复用  
- **LRU 淘汰与引用计数：** 当显存池耗尽时，依据 LRU 策略自动驱逐引用计数为 0 的叶子节点（历史对话记录）  

### Python 异步 Continuous Batching 调度器

- **双队列状态机：** 维护 `Waiting Queue`（等待 Prefill）和 `Running Queue`（正在 Decode）  
- **Iteration-level 调度：** 调度器以单个 Token 的生成为一个 Tick。每次 Tick 动态收集 `Running Queue` 中的请求，将其最新的 Token ID 拼接成二维张量，通过 PyBind11 下发给 C++ 引擎  
- **请求抢占 (Preemption)：** 当 Decode 阶段显存不足以分配新 Block 时，触发抢占逻辑，挂起优先级低的请求，将其状态从 GPU 换出（Swap-out）到 CPU，保障高优先级请求不中断  

### DeepSeek-V3 核心 CUDA 算子集

- **Fused MLA Decode Kernel：** 针对多头潜在注意力（Multi-head Latent Attention）定制算子。利用 Shared Memory 进行分块计算 (Tiling)，将 RoPE、隐变量解压和矩阵乘法融合在一个 Kernel 内，极大降低访存开销  
- **高效 Top-K 并行归约采样：** 针对最终的 Logits 向量，抛弃 CPU 端的 `np.argsort`。利用 CUB 库或纯手工编写基于 Warp/Block 级别的 Reduce 算子，在 GPU 侧直接输出下一个 Token ID  

### 高可用分布式 MoE 引擎 (Raft + NCCL)

- **数据面 (Data Plane)：** 利用 NCCL 建立多卡通信。在 MoE 模型的专家路由层，实现 `All-to-All` 集合通信，进行 Expert Parallelism (EP) 的 Token 分发与合并  
- **控制面 (Control Plane)：** 抽离出单独的 Go Raft 服务（基于 6.824 架构）。维护一张全局的 `Expert Placement Map`。当某个计算节点宕机，Raft 集群感知心跳丢失，自动触发重选举，更新并广播新的专家路由表，防止 NCCL 数据面 Hang 死  

## 目录结构

```python
Xstar-Infer/
├── xstar/                      # 引擎核心包(Python 层)
│   ├── __init__.py
│   └── layers/                 #   逐层 PyTorch module, 逐层和 HF 对拍
│       └── __init__.py
├── tests/                      # 测试
│   ├── __init__.py             #   逐层对拍 harness
│   └── harness/
│       ├── __init__.py         # 
│       └── parity_qwen2.py     # 逐层对拍 harness
├── cpp/                        # C++ 性能运行时
│   ├── tensor/                 #   裁剪 Needle → 纯前向张量库(去 autodiff)
│   ├── paged/                  #   显存池 + Radix KV cache
│   ├── kernels/                #   CUDA/Triton 算子(Fused MLA decode 等)
│   └── bindings/               #   pybind11,打通 Python 调度 ↔ C++ 计算
├── go/                         # Raft 控制面
│   └── raft/                   #   Expert Placement Map、心跳、重选举
├── configs/                    # 模型路径、block 配置等
├── bench/                      # 吞吐/延迟/roofline 脚本
├── docs/                       # 需求文档  
├── README.md
└── pyproject.toml
```

## TODO

Speculative Decoding（最推荐）：直接展示你懂"生成延迟 = memory-bound"这个本质  
Chunked Prefill / PD 解耦：和你的 PagedAttention 自然衔接  
量化（FP8/INT8 + KV cache 量化）：面试高频考点，几乎必问  

## References

[Deep Learning Systems](https://dlsyscourse.org/)

[Stanford CS336 | Language Modeling from Scratch](https://cs336.stanford.edu/)

[This is a sample AGENTS.md file from my classes](https://gist.github.com/1cg/a6c6f2276a1fe5ee172282580a44a7ac#file-agents-md)

[vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm)

[sgl-project/sglang: SGLang is a high-performance serving framework for large language models and multimodal models.](https://github.com/sgl-project/sglang)

