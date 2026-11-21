# Xstar-Infer

Xstar-Infer 是一个从零实现的单 GPU LLM 推理引擎。模型前向、CUDA 算子和分页 KV Cache 用 C++/CUDA 编写，通过 pybind11 暴露给 Python；Python 侧负责连续批处理调度，并用 FastAPI 以 SSE 流式返回生成结果。

当前支持 Qwen2.5-0.5B，BF16 推理，greedy decoding。

## 特性

- **手写 CUDA 算子**：RMSNorm、Embedding、RoPE、GEMM、融合 gate/up 投影与 SiLU 的 MLP、Softmax、FlashAttention-2 风格 attention、Paged Attention（prefill、decode、split-KV decode）。C++ 扩展只依赖 CUDA Runtime，不使用 cuBLAS、cuDNN 或 FlashAttention 库。
- **分页 KV Cache**：KV 按 16 个 token 一块管理，`BlockManager` 负责分配、引用计数和回收。
- **Radix 前缀缓存**：已完成请求的 KV 块按 token 前缀登记进 Radix 树，新请求命中前缀后直接复用这些块，只计算剩余部分。空闲前缀按 LRU 驱逐。
- **连续批处理调度**：请求随到随加入，每轮按显存余量决定接纳多少新请求，并为在跑请求预留 decode 所需的块。
- **抢占与恢复**：块不够时从运行队列尾部抢占请求，释放它的 KV；恢复时用 prompt 加已生成 token 重算。
- **Split-KV decode**：长上下文 decode 沿 KV 序列切分并行计算，再按 LSE 合并。上下文超过 4096 token 时自动启用。
- **流式 HTTP 服务**：`POST /generate`，逐 token 返回；客户端断开时调度器会回收该请求的资源。

## 快速开始

### 环境要求

- Linux，NVIDIA GPU（sm_75 及以上）
- CUDA Toolkit（含 `nvcc`）
- 支持 C++17 的编译器
- CMake 3.18 及以上
- [uv](https://docs.astral.sh/uv/)

`uv.lock` 锁定的 PyTorch 2.11.0 使用 CUDA 13 运行时，需要 580 及以上的驱动。驱动较旧时，请在 venv 中改装与驱动匹配的 CUDA 12.x 版 PyTorch。

所有命令都在仓库根目录执行。

### 1. 安装依赖并编译

```bash
git clone https://github.com/tiny-star3/Xstar-Infer.git
cd Xstar-Infer
uv sync --frozen --python 3.12

# 按本机 GPU 选择编译目标，例如 GTX 1660 Ti 为 75，A100 为 80
XSTAR_CUDA_ARCH=$(.venv/bin/python -c 'import torch; print("%d%d" % torch.cuda.get_device_capability())')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_CUDA_ARCHITECTURES="$XSTAR_CUDA_ARCH"
cmake --build build --parallel
```

编译产物输出到 `xstar_cpp_py/`。

### 2. 下载权重

服务、测试和 benchmark 都从 `~/models/Qwen2.5-0.5B` 读取权重。

```bash
.venv/bin/hf download Qwen/Qwen2.5-0.5B \
  --local-dir "$HOME/models/Qwen2.5-0.5B" \
  --include '*.json' '*.safetensors' '*.txt'
```

### 3. 启动服务

```bash
.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

### 4. 发送请求

```bash
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

## HTTP 接口

### `POST /generate`

请求体：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | 输入文本 |
| `max_tokens` | int | 30 | 最多生成的 token 数，遇到 EOS 提前结束 |
| `temperature` | float | 1.0 | 保留字段，当前未生效 |
| `top_p` | float | 1.0 | 保留字段，当前未生效 |

响应是 `text/event-stream`，每生成一个 token 发送一个事件，结束时发送 `[DONE]`：

```
data: {"token": " Paris"}

data: {"token": "."}

data: [DONE]
```

## 配置

配置项目前写在 [serve/app.py](serve/app.py) 中，修改后需重启服务：

| 配置 | 位置 | 默认值 | 说明 |
|---|---|---|---|
| 模型路径 | `qwen2_model_path` | `~/models/Qwen2.5-0.5B` | 需要 `config.json`、tokenizer 文件和单文件 `model.safetensors` |
| KV 池大小 | `num_blocks` | `ceil(max_position_embeddings / 16) * 8` | KV 块总数。Qwen2.5-0.5B 下为 16384 块，约 3 GB 显存 |
| 前缀缓存 | `Scheduler(use_radix=...)` | `True` | 关闭后每个请求都从头 prefill |

## 架构

```
HTTP 请求
  │
  ▼
serve/app.py        FastAPI，tokenize，SSE 流式返回
  │
  ▼
serve/scheduler.py  连续批处理：准入、前缀匹配、抢占、回收
  │
  ▼
serve/worker.py     拼接本轮 batch，调用前向，取 argmax
  │  pybind11
  ▼
xstar_cpp           Qwen2 前向、CUDA 算子、BlockManager、RadixTree
```

**一轮调度**

1. 处理已断开的请求，释放它们的 KV 块。
2. 依次检查等待队列：在 Radix 树中匹配最长前缀，计算还需要多少新块。块不够时先驱逐空闲前缀，仍不够就停止接纳。
3. 本轮有新请求被接纳时，对这些请求做 prefill；否则对所有在跑请求做一步 decode。同一轮不会同时做 prefill 和 decode。
4. 请求结束时，把它的 token 序列和 KV 块登记进 Radix 树，供后续请求复用。

**KV 管理**

- `BlockManager` 持有物理块池，每块保存所有层 16 个 token 的 K 和 V，按引用计数决定何时回收。
- `RadixTree` 只登记块对齐的前缀。树节点的 `lock_ref` 表示有多少在跑请求正在使用它，非零时不可驱逐。
- 块的回收由 `ref_cnt` 决定，节点能否驱逐由 `lock_ref` 决定，两者分开计数。

**Paged Attention**

- prefill：变长 batch 的 FlashAttention-2 风格 kernel，支持从已有前缀之后继续计算。
- decode：每条序列读自己的 block table。上下文超过 4096 token 时切成最多 16 段并行计算，再按 LSE 合并结果。

## 代码结构

```
xstar_cpp/
  include/, src/       C++/CUDA 核心
    ops/               算子：多数算子有一份 .cpp（CPU 实现）和一份 .cu（CUDA 实现）
    cuda/              显存分配、权重上传
    block_manager.cpp  KV 块池与引用计数
    radix_cache.cpp    Radix 前缀树
    paged_kv_cache.cpp 单个请求的 block table 与写入游标
    qwen2_model.cpp    Qwen2 前向与权重加载
bindings/              pybind11 绑定
xstar/                 PyTorch 参考实现，用作对拍基准
serve/                 HTTP 服务、调度器、worker
tests/                 测试
bench/                 性能测试脚本，说明见 bench/README.md
docs/                  各阶段的设计与调试记录
```

xstar_cpp_py/          编译产物输出目录
```

## 测试

```bash
.venv/bin/python -m pytest tests/ops      # 算子测试，不需要权重
.venv/bin/python -m pytest tests/infra    # 显存分配、BlockManager、Radix 树、权重与配置解析，部分用例需要权重
.venv/bin/python -m pytest tests/models   # 整模型测试，需要权重
```

正确性按三层对拍：

1. `xstar/` 下的 PyTorch 参考实现与 HF transformers 逐层对拍，脚本在 `tests/harness/`。
2. C++ CPU 算子与 PyTorch 参考实现对拍。
3. CUDA 算子与 CPU 算子对拍。

整模型测试把 greedy 输出与参考模型对比，并在不同执行路径之间交叉验证：非分页与分页 KV Cache、单请求与多请求批处理、启用与不启用 split-KV、完整 prefill 与基于已有前缀的增量 prefill。

BF16 下深层输出不要求逐位一致，判据见 [docs/phase0-parity-justification.md](docs/phase0-parity-justification.md)。

## Benchmark

| 脚本 | 测量内容 | 前提 |
|---|---|---|
| `bench.bench_async` | 不同并发下的吞吐和每 token 延迟 | 服务已启动 |
| `bench.bench_ttft` | 不同 prompt 长度下的首 token 延迟 | 服务已启动 |
| `bench.bench_radix_repeat` | 重复发送同一长 prompt 时，前缀缓存的收益 | 服务已启动 |
| `bench.bench_radix_multiturn` | 多轮对话中，开关前缀缓存的每轮耗时 | 分别以开、关前缀缓存启动服务各跑一次 |
| `bench.bench_splitkv` | 不同上下文长度和 batch 下，split-KV 各分片数的单步 decode 耗时 | 先停止服务以释放显存 |

运行方式为 `.venv/bin/python -m <脚本>`；`bench_splitkv` 需要调用其 `main()`。测量方法和完整结果见 [bench/README.md](bench/README.md)。

以下为 GTX 1660 Ti（6 GB）上的部分结果，测于 2026 年 8–9 月的版本：

| 场景 | 结果 |
|---|---|
| 8 个会话各 5 轮对话，第 5 轮请求耗时（关 / 开前缀缓存） | 30.96 s / 6.87 s |
| 8192 token 上下文单步 decode，16 段 split-KV 相对不切分 | batch=1 时 1.52×，batch=4 时 1.21× |
| 输出吞吐，并发 1 / 32 / 128 | 8.9 / 181.5 / 215.6 tok/s |

## 当前限制

- 只支持 Qwen2.5 架构中 `tie_word_embeddings=true` 的模型，权重须为单文件 `model.safetensors`，注意力只实现了 `head_dim=64`。目前只在 Qwen2.5-0.5B 上测试过。
- 只支持 greedy decoding，`temperature` 和 `top_p` 尚未生效。
- 流式输出逐 token 解码，一个中文字符被切成多个 token 时会显示为乱码。
- prefill 与 decode 不在同一批中执行，长 prompt 的 prefill 会推迟在跑请求的 decode。没有 chunked prefill。
- 配置写在代码中，没有命令行参数，也没有 OpenAI 兼容接口。
- 仅支持单 GPU。

## TODO

- 性能：去掉 kernel 之后的同步和每层的显存分配，GEMM 改用 tensor core，CUDA Graph。
- 正确性：直接与 HF transformers 对比 top-1 一致率和 wikitext-2 PPL；对前缀缓存和抢占做随机负载下的差分测试。
- 服务：OpenAI 兼容接口，采样，chunked prefill。
- 模型：支持 Qwen3.5 系列的混合注意力（Gated DeltaNet 与全注意力）、MoE、MTP 投机解码和 int4 量化，目标模型为 Qwen3.8-27B 和 Qwen3.6-35B-A3B。

## 参考

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [CMU Deep Learning Systems](https://dlsyscourse.org/)
- [Stanford CS336](https://cs336.stanford.edu/)
