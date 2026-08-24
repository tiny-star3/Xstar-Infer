#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "dtype.h"
#include "device.h"

/**
 * Owns OR BORROWS a contiguous (row-major) buffer of a fixed dtype/device.
 * Phase 1: CPU only, contiguous layout (no arbitrary strides).
 *   - The shape ctor (below) OWNS: allocates on construct, frees on destroy.
 *   - The (ptr, shape) ctor BORROWS: wraps external memory (e.g. a mmap view), sets owns_data_=false, and does NOT free on destroy -- the external owner is responsible.
 * Copy is disabled (would alias the buffer -> double free);
 * move transfers ownership and leaves the source empty.
 */
class Tensor
{
public:
    // Allocate an uninitialized contiguous buffer of the given shape.
    // strides_ 由 shape_ 推算(标准连续), Phase 1 不接受外部 stride
    Tensor(std::vector<std::int64_t> shape,
           DType dtype,
           Device device = Device::CPU);
    // Construct a NON-owning Tensor view over external memory at `ptr`.
    // `ptr` must stay valid for this Tensor's whole lifetime (typically ptr = mmap_base + offset).
    // On destruction this Tensor will NOT free `ptr` — the owner (e.g. MMapFile) is responsible.
    // Precondition: ptr is sufficiently aligned for dtype.
    // The view ctor itself does NOT check alignment -- it trusts the caller.
    // The real guard lives in make_weight_view, which rejects an offset that is not a multiple of dtype_size (since ptr = mmap_base + offset, alignment depends on offset, NOT on the mmap base's 4 KiB page alignment).
    // strides are computed row-major from `shape`, identical to the owned ctor.
    Tensor(const void *ptr, std::vector<std::int64_t> shape, DType dtype, Device device);

    ~Tensor();

    // 禁拷贝:两个对象指同一块 buffer 会 double-free
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    // 允移动: 转移 data_ 所有权,把 other.data_ 置 nullptr(析构时 free(nullptr) 安全)
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    // const 访问器
    const std::vector<std::int64_t> &shape() const;
    const std::vector<std::int64_t> &strides() const;
    DType dtype() const;
    Device device() const;

    // product of shape; empty shape (scalar) -> 1
    std::int64_t numel() const;
    // numel * dtype_size(dtype_)
    std::size_t nbytes() const;

    // 裸指针,给 op 内部用; op 按 dtype_ 自己 cast
    void *data();
    const void *data() const;

    // Typed access; only float and bfloat16 are specialized in the .cpp.
    // No generic definition: unsupported T fails at compile time.
    template <typename T>
    T *data();
    template <typename T>
    const T *data() const;

private:
    void *data_ = nullptr;
    std::vector<std::int64_t> shape_;
    std::vector<std::int64_t> strides_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::CPU;
    bool owns_data_;

    // free(data_) and null it; shared by dtor and move-assign
    void release();
};

/**
 * Copy a CPU Tensor's bytes to a freshly-allocated CUDA Tensor (H2D).
 *
 * Round-trip contract: to_cuda then to_cpu must reproduce the original bytes exactly (both are raw memcpy, no dtype/layout change) -- this bit-exactness is the M1 acceptance gate for "the GPU memory path works".
 *
 * Strict one-way: throws std::runtime_error if t.device() != CPU.
 * Rationale: an idempotent no-op (PyTorch's .cuda() style) would hide "you called to_cuda thinking it H2D'd, but t was already on GPU so nothing moved" -- a silent misuse.
 * M1 has no code path that legitimately to_cuda's an already-CUDA tensor; if one appears later, add an idempotent variant then.
 * Don't pre-build flexibility for a nonexistent caller.
 *
 * Returns: a new OWNED CUDA Tensor, same shape and dtype as t, device=CUDA, data = H2D copy of t's bytes.
 * t is untouched (independent memory, no aliasing).
 *
 * Preconditions: t is contiguous (owned ctor's row-major layout), t.device()==CPU.
 */
Tensor to_cuda(const Tensor &t);

/**
 * Copy a CUDA Tensor's bytes to a freshly-allocated CPU Tensor (D2H).
 * Inverse of to_cuda; strict one-way (throws if t.device() != CUDA) for the same reason.
 */
Tensor to_cpu(const Tensor &t);