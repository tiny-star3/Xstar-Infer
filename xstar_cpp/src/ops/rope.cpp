#include <stdexcept>

#include "ops/rope.h"
#include "bfloat16.h"
#include "cuda/cuda_allocator.h"

Tensor rope(const Tensor &x, const Tensor &cache, const std::int64_t *positions)
{
    if (x.shape().size() < 2 || cache.shape().size() != 3)
        throw std::runtime_error("rank mismatch");
    if (cache.shape()[0] != 2 || x.shape().back() != 2 * cache.shape()[2])
        throw std::runtime_error("shape mismatch");
    if (cache.dtype() != DType::Float32)
        throw std::runtime_error("cache dtype mismatch");
    for (size_t i = 0; i < x.shape()[x.shape().size() - 2]; i++)
        if (positions[i] < 0 || positions[i] >= cache.shape()[1])
            throw std::runtime_error("out-of-range position");
    if (x.device() != cache.device())
        throw std::runtime_error("device mismatch");
    Tensor result(x.shape(), x.dtype(), x.device());
    std::int64_t numel = x.numel();
    std::int64_t dim = x.shape().back();
    std::int64_t seq_len = x.shape()[x.shape().size() - 2];
    std::int64_t half = cache.numel() / 2;
    if (x.device() == Device::CUDA)
    {
        void *d_positions = cuda_alloc(seq_len * sizeof(std::int64_t));
        cuda_memcpy_h2d(d_positions, positions, seq_len * sizeof(std::int64_t));
        rope_launch(result.data(), x.data(), cache.data<float>(), static_cast<const int64_t *>(d_positions), numel / seq_len / dim, dim, seq_len, half, x.dtype());
        cuda_free(d_positions);

        return result;
    }
    else if (x.device() == Device::CPU)
    {
        if (x.dtype() == DType::Float32)
        {
            float *result_data = result.data<float>();
            const float *x_data = x.data<float>();
            const float *cache_data = cache.data<float>();
            for (size_t i = 0; i < numel / dim; i++)
            {
                std::int64_t p = positions[i % seq_len];
                for (size_t j = 0; j < dim / 2; j++)
                {
                    float cos = cache_data[p * (dim / 2) + j];
                    float x1 = x_data[i * dim + j];
                    float sin = cache_data[half + p * (dim / 2) + j];
                    float x2 = x_data[i * dim + dim / 2 + j];
                    result_data[i * dim + j] = cos * x1 - sin * x2;
                    result_data[i * dim + dim / 2 + j] = sin * x1 + cos * x2;
                }
            }
            return result;
        }
        else if (x.dtype() == DType::BFloat16)
        {
            bfloat16 *result_data = result.data<bfloat16>();
            const bfloat16 *x_data = x.data<bfloat16>();
            const float *cache_data = cache.data<float>();
            for (size_t i = 0; i < numel / dim; i++)
            {
                std::int64_t p = positions[i % seq_len];
                for (size_t j = 0; j < dim / 2; j++)
                {
                    bfloat16 cos = static_cast<bfloat16>(cache_data[p * (dim / 2) + j]);
                    bfloat16 x1 = x_data[i * dim + j];
                    bfloat16 sin = static_cast<bfloat16>(cache_data[half + p * (dim / 2) + j]);
                    bfloat16 x2 = x_data[i * dim + dim / 2 + j];
                    // cos * x1 - sin * x2,cos/x1/x2 都是 bfloat16。bfloat16 没有 operator*, 所以 bfloat16 * bfloat16 走 operator float() 把两边 upcast, 整个表达式在 float 里算, 最后赋值回 bfloat16 才 RNE downcast 一次
                    // 但 PyTorch CPU bf16 下,cos * x1 - sin * x2 是每步 downcast:bf16(cos*x1) 先 RNE 一次, bf16(sin*x2) 再 RNE, 最后相减再 RNE
                    // 少两次中间 downcast——精度比 PyTorch 高, 但不 bit-exact
                    result_data[i * dim + j] = cos * x1 - sin * x2;
                    result_data[i * dim + dim / 2 + j] = sin * x1 + cos * x2;
                }
            }
            return result;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}