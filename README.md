# Xstar-Infer 推理引擎

**项目定位：** 针对 DeepSeek-V4-Flash（MoE + 压缩稀疏注意力 CSA）架构的多语言推理引擎  

多语言混合架构：C++/CUDA 手写算子与显存池 + Python 异步 Continuous Batching + vLLM 部署基线  
硬件目标 2×RTX PRO 6000（96GB，Blackwell sm_120，TP=2）  

## 目标模型规格（DeepSeek-V4-Flash）

| 项           | 值                                                           |
| ------------ | ------------------------------------------------------------ |
| 总参 / 激活  | 284B / 13B                                                   |
| 层数         | 43                                                           |
| MoE          | 256 路由专家 / 6 激活 + 1 shared，moe_intermediate=2048，hidden=4096 |
| 量化         | FP8 e4m3（128×128 block scale）+ expert FP4                  |
| 注意力       | CSA（压缩稀疏），nkv=1，head_dim=512（qk_rope=64），q/o_lora_rank=1024 |
| Hash indexer | 3 层，index_topk=512                                         |
| 上下文       | 1M（YaRN factor 16，原生 64K）——部署截 32K~64K               |

## 系统架构与语言分工 (The Polyglot Architecture)

|    模块层级    |             技术栈             | 核心职责                                                     |
| :------------: | :----------------------------: | :----------------------------------------------------------- |
|  **部署基线**  |       **vLLM (≥0.28.0)**       | TP=2 跑通 V4-Flash + 调参 sweep + 理论对账（Phase 5 M1）     |
| **接入与调度** | **Python (FastAPI + asyncio)** | HTTP SSE、tokenizer、Continuous Batching（双队列 + 抢占 recompute + abort 清理） |
|  **语言绑定**  |       **C++ (pybind11)**       | Python↔C++ 边界（h/cpp/bindings 三文件契约）                 |
| **张量与内存** |            **C++**             | 纯前向张量库（去 autodiff）、mmap safetensors、BlockManager、PagedKVCache、RadixTree |
|  **MoE 算子**  |          **CUDA C++**          | router top-k → token permute → per-expert sparse GEMM → weighted combine |
| **注意力算子** |          **CUDA C++**          | 压缩稀疏注意力（CSA）kernel + hash indexer                   |
|   **分布式**   |            **NCCL**            | TP 权重分片 → EP All-to-All                                  |

## 目录结构

```text
Xstar-Infer/
├── serve/                      # 服务层(Python)
│   ├── main.py                 #   uvicorn 入口
│   ├── app.py                  #   FastAPI /generate SSE + lifespan 全局加载
│   ├── scheduler.py            #   CB 调度器(双队列 + 抢占 recompute + radix 准入 + abort 清理)
│   └── worker.py               #   run_batch(拼输入 → forward → argmax → token_queue)
├── xstar/                      # Python 参照实现(逐层与 HF 对拍)
├── xstar_cpp/                  # C++/CUDA 引擎
│   ├── include/                #   tensor / kv_cache / paged_kv_cache / block_manager / radix_cache / ops/*
│   └── src/                    #   每 op .cpp(CPU) + .cu(CUDA)；paged_attention 拆 decode/splitKV/prefill
│       └── moe*.cu / csa*.cu
├── bindings/python_bindings.cpp  # pybind11 绑定
├── xstar_cpp_py/               # 编译产物(.so)
├── tests/                      # 一 op 一文件(bridge / harness / infra / ops / models / serve)
├── bench/                      # 吞吐 / TTFT / radix / splitKV 脚本
│   └── bench_vllm/      # vLLM 调参 sweep
├── deploy/                     # (计划) vLLM 启动脚本 + 调参记录
├── docs/                       # 每 M 阶段 takeaway + 模拟面试
└── CMakeLists.txt
```

## Bench（数据待填）

| 项                            | 数值     |
| ----------------------------- | -------- |
| vLLM TP=2 吞吐                | （待填） |
| vLLM TP=2 TTFT                | （待填） |
| 调参 sweep 最优组合           | （待填） |
| 理论 decode 上界 vs 实测      | （待填） |
| MoE 接入后引擎吞吐            | （待填） |
| CSA kernel vs dense attention | （待填） |

## TODO

Speculative Decoding
Chunked Prefill / PD 解耦 
量化（FP8/INT8 + KV cache 量化） 

## References

[Deep Learning Systems](https://dlsyscourse.org/)

[Stanford CS336 | Language Modeling from Scratch](https://cs336.stanford.edu/)

[This is a sample AGENTS.md file from my classes](https://gist.github.com/1cg/a6c6f2276a1fe5ee172282580a44a7ac#file-agents-md)

[vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm)

[sgl-project/sglang: SGLang is a high-performance serving framework for large language models and multimodal models.](https://github.com/sgl-project/sglang)

