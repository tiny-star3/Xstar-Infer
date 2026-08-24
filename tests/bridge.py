import pytest
import sys
import numpy as np
import torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


def torch_to_cpp(t: torch.Tensor) -> xstar_cpp.Tensor:
    """
    Copy a torch tensor's raw bytes into an xstar Tensor, bit-exact.

    BFloat16 is reinterpreted as uint16 (NOT downcast to float16):
        bf16 and f16 are different formats, so the only bit-exact crossing is to view the 16 bf16 bits as uint16 and hand xstar the raw bytes (xstar reinterprets them back as BFloat16 via from_numpy_raw).
        Float32 copies straight through.

    This is the gate of all parity judgment: if this round-trip is not bit-exact, every downstream allclose is measuring bridge error, not C++ correctness.

    Args:
        t: a C-contiguous torch tensor of dtype float32 or bfloat16.
            Non-contiguous inputs raise (torch .view(uint16) requires contiguity).

    Returns:
        An xstar Tensor with the same shape and matching dtype (Float32/BFloat16),
        owning a copy of t's bytes.
    """
    if t.dtype == torch.float32:
        return xstar_cpp.from_numpy_raw(t.numpy(), t.shape, xstar_cpp.DType.Float32)
    elif t.dtype == torch.bfloat16:
        return xstar_cpp.from_numpy_raw(
            t.view(torch.uint16).numpy(), t.shape, xstar_cpp.DType.BFloat16
        )
    else:
        raise RuntimeError("unsupported dtype")


def cpp_to_torch(t: xstar_cpp.Tensor, ref_shape) -> torch.Tensor:
    """
    Copy an xstar Tensor's raw bytes back to torch, bit-exact (inverse of torch_to_cpp).

    The xstar Tensor comes out as a FLAT uint8 byte buffer (to_numpy_raw); ref_shape is required because the buffer carries no shape.
    BFloat16 is reinterpreted uint16->bf16 (the inverse of the uint16 view on the way in).
    Float32 copies straight through.

    CAUTION: ref_shape is trusted -- if the C++ tensor's true shape has the same element count but different extents, the reshape silently succeeds and hides the bug.
    Callers should assert cpp shape == ref shape BEFORE calling this (see test_cpp_qwen2_model.py for the pattern).

    Args:
        t: an xstar Tensor of dtype Float32 or BFloat16.
        ref_shape: the intended output shape (element count must match t's).

    Returns:
        A torch tensor of shape ref_shape, dtype float32 or bfloat16, owning a copy.
    """
    if t.dtype() == xstar_cpp.DType.Float32:
        t_np = xstar_cpp.to_numpy_raw(t).view(np.float32).reshape(ref_shape)
        return torch.from_numpy(t_np)
    elif t.dtype() == xstar_cpp.DType.BFloat16:
        t_np = xstar_cpp.to_numpy_raw(t).view(np.uint16).reshape(ref_shape)
        return torch.from_numpy(t_np).view(torch.bfloat16)
    else:
        raise RuntimeError("unsupported dtype")
