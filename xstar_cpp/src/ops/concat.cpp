#include <stdexcept>

#include "ops/concat.h"
#include "cuda/cuda_allocator.h"

Tensor concat(const std::vector<const Tensor *> &inputs, int axis)
{
    if (inputs.empty())
        throw std::runtime_error("empty inputs");
    std::size_t rank = inputs[0]->shape().size();
    if (rank > MAX_RANK)
        throw std::runtime_error("inputs rank exceed MAX_RANK");
    DType dtype = inputs[0]->dtype();
    Device device = inputs[0]->device();
    if (device != Device::CUDA)
        throw std::runtime_error("concat is GPU-only");
    size_t n = inputs.size();
    for (std::size_t i = 0; i < n; i++)
    {
        if (inputs[i]->shape().size() != rank)
            throw std::runtime_error("rank mismatch");
        if (inputs[i]->dtype() != dtype)
            throw std::runtime_error("dtype mismatch");
        if (inputs[i]->device() != device)
            throw std::runtime_error("device mismatch");
    }
    // 当 signed 和 unsigned 同阶(都是 64-bit)比较时, signed 那个被转成 unsigned
    // 无符号类型参与有符号比较时, 比较语义会被隐式重写, 手动转化 int64_t
    if (axis < -static_cast<int64_t>(rank) || axis >= static_cast<int64_t>(rank))
        throw std::runtime_error("axis out of [-rank, rank)");
    if (axis < 0)
        axis += rank;
    std::int64_t axis_dim = 0;
    for (std::size_t i = 0; i < rank; i++)
    {
        if (i == axis)
        {
            for (std::size_t j = 0; j < n; j++)
            {
                axis_dim += inputs[j]->shape()[axis];
            }
            continue;
        }
        std::int64_t dim = inputs[0]->shape()[i];
        for (std::size_t j = 0; j < n; j++)
        {
            if (inputs[j]->shape()[i] != dim)
                throw std::runtime_error("non-axis dim mismatch");
        }
    }

    std::vector<std::int64_t> result_shape(inputs[0]->shape());
    result_shape[axis] = axis_dim;
    Tensor result(result_shape, dtype, device);
    std::vector<const void *> inputs_ptrs;
    std::vector<std::int64_t> inputs_axis_sizes;
    for (std::size_t i = 0; i < n; i++)
    {
        inputs_ptrs.push_back(inputs[i]->data());
        inputs_axis_sizes.push_back(inputs[i]->shape()[axis]);
    }
    void *d_ptrs = cuda_alloc(n * sizeof(const void *));
    cuda_memcpy_h2d(d_ptrs, inputs_ptrs.data(), n * sizeof(const void *));
    void *d_axis_sizes = cuda_alloc(n * sizeof(const std::int64_t));
    cuda_memcpy_h2d(d_axis_sizes, inputs_axis_sizes.data(), n * sizeof(const std::int64_t));
    void *d_out_shape = cuda_alloc(rank * sizeof(const std::int64_t));
    cuda_memcpy_h2d(d_out_shape, result.shape().data(), rank * sizeof(const std::int64_t));

    concat_launch(result.data(), static_cast<void **>(d_ptrs), static_cast<const std::int64_t *>(d_axis_sizes), static_cast<const int64_t *>(d_out_shape), n, rank, axis, dtype_size(dtype), result.numel());

    cuda_free(d_ptrs);
    cuda_free(d_axis_sizes);
    cuda_free(d_out_shape);
    return result;
}