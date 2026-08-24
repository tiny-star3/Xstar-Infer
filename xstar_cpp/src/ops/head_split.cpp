#include <cstring>

#include "ops/head_split.h"

Tensor head_split(const Tensor &t, std::int64_t heads)
{
    if (t.shape().size() != 2)
        throw std::runtime_error("rank mismatch");
    if (t.dtype() != DType::Float32 && t.dtype() != DType::BFloat16)
        throw std::runtime_error("dtype unsupported");
    if (heads <= 0)
        throw std::runtime_error("heads not positive");
    if (t.shape()[1] % heads != 0)
        throw std::runtime_error("head_dim not integral");

    std::vector<std::int64_t> result_shape;
    std::int64_t seq = t.shape()[t.shape().size() - 2];
    std::int64_t head_dim = t.shape()[t.shape().size() - 1] / heads;
    for (size_t i = 0; i < t.shape().size() - 2; i++)
    {
        result_shape.push_back(t.shape()[i]);
    }
    result_shape.push_back(heads);
    result_shape.push_back(seq);
    result_shape.push_back(head_dim);
    Tensor result(result_shape, t.dtype(), t.device());
    if (t.device() == Device::CUDA)
    {
        head_split_launch(result.data(), t.data(), heads, seq, head_dim, t.dtype());

        return result;
    }
    else if (t.device() == Device::CPU)
    {
        for (std::int64_t i = 0; i < heads; i++)
        {
            for (std::int64_t j = 0; j < seq; j++)
            {
                memcpy(static_cast<void *>(static_cast<char *>(result.data()) + (i * seq * head_dim + j * head_dim) * dtype_size(t.dtype())), static_cast<const void *>(static_cast<const char *>(t.data()) + (j * heads * head_dim + i * head_dim) * dtype_size(t.dtype())), head_dim * dtype_size(t.dtype()));
            }
        }

        return result;
    }
    else
        throw std::runtime_error("unsupported device");
}