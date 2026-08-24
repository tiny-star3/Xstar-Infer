import struct
import json
import pytest
import os
import sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp

qwen2_model_path = "~/models/Qwen2.5-0.5B/model.safetensors"
qwen2_model_path = os.path.expanduser(qwen2_model_path)


def _make_safetensors(tensors, metadata, path):
    """
    Build a minimal safetensors file at `path` for error-branch tests.

    tensors: dict name -> {"dtype": "F32"|"BF16", "shape": [int, ...], "data": bytes}.
    metadata: dict or None -> written as the __metadata__ key (None omits it).
    Returns `path`.

    This is a TEST FIXTURE (data preparation via stdlib struct/json), not parser logic:
        it assembles the [8B header_len][JSON header][data] byte layout so the C++ parser under test receives a well-formed (or deliberately malformed) safetensors file.
    """
    header = {}
    offset = 0
    data = b""
    for name, t in tensors.items():
        nbytes = len(t["data"])
        header[name] = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        data += t["data"]
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode()
    # safetensors 规范: header 补空格使 (8+len)%8==0, 数据段对齐否则 make_weight_view 报 offset not aligned
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)
    return path


# 解析器读到的 key 集合和 safe_open 完全一致,不多不少
def test_parse_returns_all_keys():
    mf = xstar_cpp.MMapFile(qwen2_model_path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    assert set(meta.keys()) == set(safe_open(qwen2_model_path, framework="pt").keys())


# embed 张量的 shape/dtype/offset 三者都正确,且 offset 喂 make_weight_view 切出的字节和 safe_open bit-exact
def test_parse_embed_bit_exact():
    mf = xstar_cpp.MMapFile(qwen2_model_path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    k = "model.embed_tokens.weight"
    view = xstar_cpp.make_weight_view(
        mf,
        meta[k].offset,
        meta[k].shape,
        meta[k].dtype,
    )
    cpp_bytes = xstar_cpp.to_numpy_raw(view)

    with safe_open(qwen2_model_path, framework="pt") as f:
        ref = f.get_tensor(k)
        # bf16 当裸字节
        ref_bytes = ref.contiguous().view(torch.uint8).numpy().ravel()
        # shape 对
        assert tuple(meta[k].shape) == tuple(ref.shape)
        # dtype 对
        assert {
            xstar_cpp.DType.BFloat16: torch.bfloat16,
            xstar_cpp.DType.Float32: torch.float32,
        }[meta[k].dtype] == ref.dtype
        # 字节 bit-exact
        assert np.array_equal(ref_bytes, cpp_bytes)
        # nbytes 对
        assert len(cpp_bytes) == ref.numel() * ref.element_size()


# 中间层(非 0 非 23)的 weight 路径。和 embed 同结构,但 k 用 model.layers.10.self_attn.q_proj.weight——证明不是只有 layer 0 / embed 能过
def test_parse_mid_layer_weight_bit_exact():
    mf = xstar_cpp.MMapFile(qwen2_model_path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    k = "model.layers.10.self_attn.q_proj.weight"
    view = xstar_cpp.make_weight_view(
        mf,
        meta[k].offset,
        meta[k].shape,
        meta[k].dtype,
    )
    cpp_bytes = xstar_cpp.to_numpy_raw(view)

    with safe_open(qwen2_model_path, framework="pt") as f:
        ref = f.get_tensor(k)
        # bf16 当裸字节
        ref_bytes = ref.contiguous().view(torch.uint8).numpy().ravel()
        # shape 对
        assert tuple(meta[k].shape) == tuple(ref.shape)
        # dtype 对
        assert {
            xstar_cpp.DType.BFloat16: torch.bfloat16,
            xstar_cpp.DType.Float32: torch.float32,
        }[meta[k].dtype] == ref.dtype
        # 字节 bit-exact
        assert np.array_equal(ref_bytes, cpp_bytes)
        # nbytes 对
        assert len(cpp_bytes) == ref.numel() * ref.element_size()


# bias 张量(1-D,shape [896])路径。bias 和 weight 的 shape 维度不同(1-D vs 2-D),分开测避免"只测了 2-D"盲区
def test_parse_q_proj_bias_bit_exact():
    mf = xstar_cpp.MMapFile(qwen2_model_path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    k = "model.layers.10.self_attn.q_proj.bias"
    view = xstar_cpp.make_weight_view(
        mf,
        meta[k].offset,
        meta[k].shape,
        meta[k].dtype,
    )
    cpp_bytes = xstar_cpp.to_numpy_raw(view)

    with safe_open(qwen2_model_path, framework="pt") as f:
        ref = f.get_tensor(k)
        # bf16 当裸字节
        ref_bytes = ref.contiguous().view(torch.uint8).numpy().ravel()
        # shape 对
        assert tuple(meta[k].shape) == tuple(ref.shape)
        # dtype 对
        assert {
            xstar_cpp.DType.BFloat16: torch.bfloat16,
            xstar_cpp.DType.Float32: torch.float32,
        }[meta[k].dtype] == ref.dtype
        # 字节 bit-exact
        assert np.array_equal(ref_bytes, cpp_bytes)
        # nbytes 对
        assert len(cpp_bytes) == ref.numel() * ref.element_size()


# o_proj 只有 weight 没 bias, 且这个 weight 能正确解析
def test_parse_o_proj_no_bias():
    mf = xstar_cpp.MMapFile(qwen2_model_path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    k = "model.layers.23.self_attn.o_proj.weight"
    view = xstar_cpp.make_weight_view(
        mf,
        meta[k].offset,
        meta[k].shape,
        meta[k].dtype,
    )
    cpp_bytes = xstar_cpp.to_numpy_raw(view)

    with safe_open(qwen2_model_path, framework="pt") as f:
        ref = f.get_tensor(k)
        # bf16 当裸字节
        ref_bytes = ref.contiguous().view(torch.uint8).numpy().ravel()
        # shape 对
        assert tuple(meta[k].shape) == tuple(ref.shape)
        # dtype 对
        assert {
            xstar_cpp.DType.BFloat16: torch.bfloat16,
            xstar_cpp.DType.Float32: torch.float32,
        }[meta[k].dtype] == ref.dtype
        # 字节 bit-exact
        assert np.array_equal(ref_bytes, cpp_bytes)
        # nbytes 对
        assert len(cpp_bytes) == ref.numel() * ref.element_size()


# model.norm.weight(final RMSNorm,非层内)路径。和 embed 同属"非层 key",但 shape 不同(1-D [896] vs 2-D)
def test_parse_final_norm_bit_exact():
    mf = xstar_cpp.MMapFile(qwen2_model_path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    k = "model.norm.weight"
    view = xstar_cpp.make_weight_view(
        mf,
        meta[k].offset,
        meta[k].shape,
        meta[k].dtype,
    )
    cpp_bytes = xstar_cpp.to_numpy_raw(view)

    with safe_open(qwen2_model_path, framework="pt") as f:
        ref = f.get_tensor(k)
        # bf16 当裸字节
        ref_bytes = ref.contiguous().view(torch.uint8).numpy().ravel()
        # shape 对
        assert tuple(meta[k].shape) == tuple(ref.shape)
        # dtype 对
        assert {
            xstar_cpp.DType.BFloat16: torch.bfloat16,
            xstar_cpp.DType.Float32: torch.float32,
        }[meta[k].dtype] == ref.dtype
        # 字节 bit-exact
        assert np.array_equal(ref_bytes, cpp_bytes)
        # nbytes 对
        assert len(cpp_bytes) == ref.numel() * ref.element_size()


# 文件 < 8 字节,连 header_len 都读不到
def test_parse_rejects_file_too_small(tmp_path):
    path = str(tmp_path / "x")
    with open(path, "wb") as f:
        f.write(b"1234")
    mf = xstar_cpp.MMapFile(path)
    with pytest.raises(RuntimeError, match="file smaller than 8 bytes"):
        xstar_cpp.parse_safetensors_header(mf)


# 8 字节 header_len 声称 header 有 N 字节,但文件总大小 < 8+N(header 被截断)
def test_parse_rejects_truncated_header(tmp_path):
    path = str(tmp_path / "x")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1000))
        f.write(b"0123456789")
    mf = xstar_cpp.MMapFile(path)
    with pytest.raises(RuntimeError, match="truncated header"):
        xstar_cpp.parse_safetensors_header(mf)


# header 里某 tensor 的 dtype 是 "F16",不在 F32/BF16 之列
def test_parse_rejects_unsupported_dtype(tmp_path):
    path = str(tmp_path / "x")
    path = _make_safetensors(
        {"y": {"dtype": "F16", "shape": [1], "data": b"\x00\x00"}}, None, path
    )
    mf = xstar_cpp.MMapFile(path)
    with pytest.raises(RuntimeError, match="dtype not"):
        xstar_cpp.parse_safetensors_header(mf)


# __metadata__ 键被跳过,不进 map、不报错,其他 tensor 正常解析
def test_parse_skips_metadata_key(tmp_path):
    path = str(tmp_path / "x")
    path = _make_safetensors(
        {
            "x": {
                "dtype": "F32",
                "shape": [1],
                "data": np.array([1.0], dtype=np.float32).tobytes(),
            }
        },
        {"format": "pt"},
        path,
    )
    mf = xstar_cpp.MMapFile(path)
    meta = xstar_cpp.parse_safetensors_header(mf)
    assert "__metadata__" not in meta and "x" in meta and list(meta["x"].shape) == [1]


# shape 和 data_offsets 不自洽——shape 说 4 字节,data_offsets 只给 2 字节
def test_parse_rejects_inconsistent_header(tmp_path):
    path = str(tmp_path / "x")
    path = _make_safetensors(
        {"z": {"dtype": "F32", "shape": [2], "data": b"\x00\x00"}}, None, path
    )
    mf = xstar_cpp.MMapFile(path)
    with pytest.raises(RuntimeError, match="header inconsistent"):
        xstar_cpp.parse_safetensors_header(mf)
