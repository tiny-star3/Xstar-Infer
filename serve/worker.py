import sys
from starlette.concurrency import run_in_threadpool
from tests.bridge import cpp_to_torch
import torch
from serve.scheduler import State
import numpy as np

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


class Worker:

    def __init__(self, weights, cfg, rope, bm, block_size, dtype, device):
        self.weights = weights
        self.cfg = cfg
        self.rope = rope
        self.bm = bm
        # 预存 new PagedKVCache 要的派生量(省每次算)
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.nkv = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.max_position_embeddings

    async def run_batch(self, reqs, is_decode):
        #  对一批请求执行一次 forward,分发每请求的 next token
        input_ids = []
        cu_seqlens = [0]
        if is_decode:
            input_ids = np.array([req.generated_ids[-1] for req in reqs])
            cu_seqlens = list(range(len(reqs) + 1))
        else:
            # 可能是 re-prefill 之前已经计算的部分 generated_ids
            # 去掉前缀共享
            seqs = [
                (req.prompt_ids + req.generated_ids)[req.kv.cursor() :] for req in reqs
            ]
            input_ids = np.concatenate([np.array(s, dtype=np.int64) for s in seqs])
            for s in seqs:
                cu_seqlens.append(cu_seqlens[-1] + len(s))

        kv_caches = [req.kv for req in reqs]

        # 调 forward(线程池)
        logits_cuda = await run_in_threadpool(
            xstar_cpp.qwen2_forward_multi,
            self.weights,
            self.cfg,
            self.rope,
            self.bm,
            kv_caches,
            is_decode,
            input_ids,
            cu_seqlens,
        )
        logits = cpp_to_torch(
            xstar_cpp.to_cpu(logits_cuda), [len(input_ids), self.cfg.vocab_size]
        )

        # 逐请求取 argmax + 放 queue
        if is_decode:
            tokens = logits.argmax(-1)
        else:
            last_rows = torch.tensor(cu_seqlens[1:]) - 1
            tokens = logits[last_rows].argmax(-1)

        for req, token in zip(reqs, tokens):
            req.generated_ids.append(int(token))
            req.token_queue.put_nowait(int(token))

            # 状态转移 + 判终止
            if not is_decode:
                req.state = State.RUNNING

            if (
                int(token) == req.eos_token_id
                or len(req.generated_ids) >= req.max_tokens
            ):
                req.state = State.FINISHED
                # 结束信号
                req.token_queue.put_nowait(None)
