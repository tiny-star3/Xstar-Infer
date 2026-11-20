# Xstar-Infer

Xstar-Infer 是一个面向 **Qwen2.5-0.5B 的单 GPU 推理引擎**。C++/CUDA 实现模型前向、注意力算子和分页 KV Cache，Python 调度并发请求，通过 FastAPI 逐 token 返回结果。可以在同一套代码里观察三个问题：多轮对话能复用多少历史计算，显存紧张时如何恢复被抢占的请求，以及长上下文解码何时值得沿 KV 序列分片。

当前服务使用 **BF16、greedy decoding、每块 16 个 token**。从 [请求调度](serve/scheduler.py)、[块与前缀管理](xstar_cpp/src/radix_cache.cpp) 或 [Split-KV CUDA 实现](xstar_cpp/src/ops/paged_attention_decode_split.cu) 开始阅读。

## 实验结果

下表摘自仓库内的单卡历史实验，模型为 Qwen2.5-0.5B，使用 BF16 和 6 GB 显存的 GPU。原记录未完整列出每组实验的构建提交、驱动和 CUDA 版本，数值尚未在当前提交上独立复测。

| 问题                           | 工作负载和对照                                               | 历史结果                                                     |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 复用对话历史能省多少时间？     | 8 个并发会话，各自连续对话 5 轮，每轮最多生成 16 个 token。相同脚本分别关闭、开启 Radix 缓存 | 第 1 轮请求完成耗时为 5.88 s / 5.94 s，第 5 轮为 **30.96 s / 6.87 s**。计时截止到整条响应结束。[脚本](bench/bench_radix_multiturn.py) · [记录](bench/README.md) |
| Split-KV 在什么负载下有收益？  | 上下文 8,192 tokens，16 路切分与单路对照。用 CUDA Event 计时整模型单步 decode forward | batch=1 时加速 **1.52×**，batch=4 时为 **1.21×**。相同上下文下，收益随 batch 改变。[脚本](bench/bench_splitkv.py) · [记录](docs/phase4-m3-takeaway.md) |
| 提高客户端并发能换来多少吞吐？ | 同一短 prompt，每请求最多生成 64 个 token，32 路与 128 路并发各跑一次 | 总输出吞吐分别为 **181.5、215.6 tok/s**。并发升至 4 倍，吞吐增加约 19%。[脚本](bench/bench_async.py) · [记录](bench/README.md) |

多轮脚本的历史输出列名是 `ttft_median`，实际计入了完整生成过程。并发脚本的 `tpot` 包含首 token 前的耗时。若要测请求级 TTFT 和 TPOT，需要分别记录请求发出、首 token 和末 token 的时间。

## 请求如何执行

1. **批量前向。** [HTTP 服务](serve/app.py) 把输入编码后交给调度器。[Worker](serve/worker.py) 拼接同一轮的变长输入，通过 [pybind11](bindings/python_bindings.cpp) 调用 C++ 模型前向，再把选出的 token 分发到各请求的队列。调度器优先处理本轮获准进入的 prefill 请求，其余轮次推进 decode。
2. **前缀复用与抢占。** [BlockManager](xstar_cpp/src/block_manager.cpp) 管理物理 KV 块，[RadixTree](xstar_cpp/src/radix_cache.cpp) 索引可复用的前缀。命中时，请求接入已有块，从 KV 游标之后开始计算。可用块不足时，调度器从运行队列尾部抢占请求，释放其持有的 KV 引用；恢复时按原 prompt 和已生成 token 重建所需 KV。[调度实现](serve/scheduler.py) · [块引用与驱逐测试](tests/infra/test_cpp_radix_ownership_closure.py)
3. **长上下文解码。** [Paged Attention](xstar_cpp/src/ops/paged_attention.cpp) 选择 prefill 或 decode 路径。Split-KV 沿序列维度并行计算各段的局部结果，再按在线 softmax 的归一化统计量合并。额外的 kernel 启动和临时缓冲有成本，因此分片数可以单独对照。[CUDA 算子](xstar_cpp/src/ops/paged_attention_decode_split.cu) · [实验记录](docs/phase4-m3-takeaway.md)

## 运行第一个请求

需要 Linux、带兼容驱动的 NVIDIA GPU、CUDA Toolkit（含 `nvcc`）、支持 C++17 的编译器、支持 CUDA 架构选择的 CMake 和 uv。Python 版本及依赖见 [pyproject.toml](pyproject.toml) 和 [uv.lock](uv.lock)。克隆仓库后，从仓库根目录执行构建和启动命令。

**1. 安装依赖并编译扩展**

```bash
git clone https://github.com/tiny-star3/Xstar-Infer.git
cd Xstar-Infer
uv sync --frozen --python 3.12

# 在用于运行服务的单卡机器上，按实际 GPU 选择编译目标。
XSTAR_CUDA_ARCH=$(.venv/bin/python -c 'import torch; print("%d%d" % torch.cuda.get_device_capability())')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_CUDA_ARCHITECTURES="$XSTAR_CUDA_ARCH"
cmake --build build --parallel
```

**2. 准备权重并启动服务**

[服务入口](serve/app.py) 从 `~/models/Qwen2.5-0.5B` 读取配置、tokenizer 和单文件 `model.safetensors`。已有完整权重时可跳过下载。

```bash
.venv/bin/hf download Qwen/Qwen2.5-0.5B \
  --local-dir "$HOME/models/Qwen2.5-0.5B" \
  --include '*.json' '*.safetensors' '*.txt'

.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

**3. 在另一个终端请求流式生成**

```bash
curl -N --max-time 120 http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16}'
```

每个 SSE 事件的格式为 `data: {"token": "..."}`，正常完成后收到 `data: [DONE]`。当前服务按 argmax 选取下一个 token。显存预算由 [serve/app.py](serve/app.py) 中的 `num_blocks` 控制，调整后需重启服务。

## 验证与复现实验

| 任务                    | 从仓库根目录运行                                             | 条件                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 检查 Radix 块引用与驱逐 | `.venv/bin/python -m pytest tests/infra/test_cpp_radix_ownership_closure.py` | 编译好扩展，有 GPU 即可                                      |
| 检查 Split-KV 解码      | `.venv/bin/python -m pytest tests/models/test_cpp_splitkv_decode.py` | 需要 GPU、扩展和模型权重                                     |
| 比较多轮对话缓存开关    | `.venv/bin/python -m bench.bench_radix_multiturn`            | 在服务入口分别设 `use_radix=False` / `True`，每次重启后跑相同客户端 |
| 测并发与输出吞吐        | `.venv/bin/python -m bench.bench_async`                      | 先启动 HTTP 服务                                             |

单独复现上表的 Split-KV 对照时，先停掉 HTTP 服务以释放显存。脚本提供 `main()` 函数，以下命令只测 8,192 tokens，输出各分片数相对单路的结果。

```bash
.venv/bin/python - <<'PY'
from bench import bench_splitkv

bench_splitkv.CONTEXTS = [8192]
bench_splitkv.BATCHES = [1, 4]
bench_splitkv.main()
PY
```

更多实现与对拍入口见 [CUDA 算子](xstar_cpp/src/ops/)、[PyTorch 参考模型](xstar/models/qwen2.py) 和 [模型测试](tests/models/)。现有参考流程将 Hugging Face 权重加载到仓库自有 PyTorch 实现中；独立 Transformers logits 对照仍是待补的验证项。

## 当前范围与下一步

当前验证范围集中在单卡 Qwen2.5-0.5B，分页注意力主路径使用 `head_dim=64`。下一步优先补齐三个环节：

- 检查前缀在块内分叉、请求完成和取消时的 KV 有效长度与引用计数。
- 在流式客户端记录逐 token 时间，使用相同输入、并发和缓存状态对照 TTFT、TPOT 与请求完成耗时。
- 在长短请求混合的负载下比较现有 prefill 优先策略与分块 prefill 的尾延迟。

参考：[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[Deep Learning Systems](https://dlsyscourse.org/) 和 [Stanford CS336](https://cs336.stanford.edu/)。
