from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import os
import sys
import math
import transformers
from tests.bridge import torch_to_cpp
from xstar.layers.rope import RoPE
import json
from serve.scheduler import Scheduler, Request
from serve.worker import Worker

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp

# 本地模型路径
qwen2_model_path = "~/models/Qwen2.5-0.5B"
qwen2_model_path = os.path.expanduser(qwen2_model_path)


# 全局初始化(启动时 load 一次)
# lifespan 在 app 启动时跑 yield 前的代码,load 全局;关闭时跑 yield 后
@asynccontextmanager
async def lifespan(app):
    # 启动: load 全局
    global dtype, device, weights, cfg, rope_cache_cuda, tokenizer, eos_token_id, block_size, bm, scheduler

    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA
    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache = RoPE(
        cfg.rope_theta, head_dim, cfg.max_position_embeddings, device="cuda"
    )._freq_cis_cache
    rope_cache_cuda = xstar_cpp.to_cuda(torch_to_cpp(cache.cpu()))
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    weights = xstar_cpp.load_qwen2_weights(mf, cfg, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        qwen2_model_path,
    )
    eos_token_id = tokenizer.eos_token_id

    block_size = 16
    nkv = cfg.num_key_value_heads
    dtype_size = 2
    kv_slot_bytes = nkv * head_dim * dtype_size * 2
    num_layers = cfg.num_hidden_layers
    max_seq_len = cfg.max_position_embeddings
    num_blocks = math.ceil(max_seq_len / block_size) * 8

    bm = xstar_cpp.BlockManager(
        num_blocks, block_size, kv_slot_bytes, device, num_layers
    )
    scheduler = Scheduler(
        Worker(weights, cfg, rope_cache_cuda, bm, block_size, dtype, device)
    )
    scheduler.start()

    yield
    # 关闭(可选)
    scheduler._task.cancel()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 30
    temperature: float = 1.0
    top_p: float = 1.0


# 流式响应(StreamingResponse + SSE)
# FastAPI 自动把 POST body 的 JSON 解析成 GenerateRequest, req.prompt 直接可用
# StreamingResponse 吃一个 async generator(带 yield 的 async def), 把每次 yield 的字节流式返回
# media_type="text/event-stream" 让客户端按 SSE 解析
@app.post("/generate")
async def generate(req: GenerateRequest):
    return StreamingResponse(stream(req), media_type="text/event-stream")


# 把 token 文本拼成 SSE 格式
def sse(token):
    return "data: " + json.dumps({"token": token}, ensure_ascii=False) + "\n\n"


# async generator(SSE 格式)
# 每个 event 是 data: {json}\n\n(结尾两个 \n)
# SSE 协议,客户端(浏览器/curl)按这个分 event
async def stream(req):
    ids = tokenizer(req.prompt, return_tensors="pt").input_ids.tolist()[0]

    request = Request(ids, req.max_tokens, eos_token_id)

    scheduler.submit(request)

    # 消费 queue
    while True:
        token = await request.token_queue.get()
        if token is None:
            break
        yield sse(tokenizer.decode(token))

    yield "data: [DONE]\n\n"
