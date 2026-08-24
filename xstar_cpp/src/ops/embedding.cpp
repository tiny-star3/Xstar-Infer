#include <cstring>

#include "ops/embedding.h"
#include "cuda/cuda_allocator.h"

Tensor embedding(const Tensor &weight, const std::int64_t *ids, const std::vector<std::int64_t> &ids_shape)
{
    if (weight.shape().size() != 2)
        throw std::runtime_error("weight not 2-D");
    if (weight.dtype() != DType::Float32 && weight.dtype() != DType::BFloat16)
        throw std::runtime_error("weight not a float dtype");
    if (ids_shape.size() < 1)
        throw std::runtime_error("empty ids_shape");

    std::int64_t numel = 1;
    std::int64_t vocab_size = weight.shape()[0];
    std::int64_t hidden = weight.shape()[1];
    for (auto shape : ids_shape)
        numel *= shape;
    for (std::int64_t i = 0; i < numel; i++)
        if (ids[i] < 0 || ids[i] >= vocab_size)
            throw std::runtime_error("out-of-range index");

    std::vector<std::int64_t> out_shape(ids_shape);
    out_shape.push_back(hidden);
    Tensor result(out_shape, weight.dtype(), weight.device());

    if (weight.device() == Device::CUDA)
    {
        void *d_ids = cuda_alloc(numel * sizeof(std::int64_t));
        cuda_memcpy_h2d(d_ids, ids, numel * sizeof(std::int64_t));
        embedding_launch(result.data(), weight.data(), static_cast<int64_t *>(d_ids), numel, hidden, weight.dtype());
        cuda_free(d_ids);

        return result;
    }
    else if (weight.device() == Device::CPU)
    {
        char *result_data = static_cast<char *>(result.data());
        const char *weight_data = static_cast<const char *>(weight.data());
        size_t row_bytes = hidden * dtype_size(weight.dtype());
        for (std::int64_t i = 0; i < numel; i++)
        {
            memcpy(result_data + i * row_bytes, weight_data + ids[i] * row_bytes, row_bytes);
        }
        return result;
    }
    else
        throw std::runtime_error("unsupported device");
}
