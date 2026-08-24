#include <cmath>

#include "ops/rmsnorm.h"
#include "bfloat16.h"

Tensor rmsnorm(const Tensor &x, const Tensor &weight, float eps)
{
    // f32 归一化（乘 rms）→ 转回 bf16 → 乘 bf16 weight
    if (x.dtype() != weight.dtype())
        throw std::runtime_error("rmsnorm: x and weight must have the same dtype");
    if (x.shape().size() < 1 || weight.shape().size() != 1)
        throw std::runtime_error("rmsnorm expects x rank >= 1 and weight 1D");
    if (x.shape().back() != weight.shape()[0])
        throw std::runtime_error("shape mismatch");
    if (x.device() != weight.device())
        throw std::runtime_error("device mismatch");

    size_t hidden = x.shape().back();
    size_t num_rows = x.numel() / hidden;
    Tensor result(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::CUDA)
    {
        rmsnorm_launch(result.data(), x.data(), weight.data(), static_cast<std::int64_t>(hidden), static_cast<std::int64_t>(num_rows), eps, x.dtype());

        return result;
    }
    else if (x.device() == Device::CPU)
    {
        if (x.dtype() == DType::Float32)
        {
            const float *x_data = x.data<float>();
            const float *weight_data = weight.data<float>();
            float *result_data = result.data<float>();
            for (size_t i = 0; i < num_rows; i++)
            {
                float rms = 0;
                for (size_t j = 0; j < hidden; j++)
                {
                    float xij = static_cast<float>(x_data[i * hidden + j]);
                    rms += xij * xij;
                }
                // 故意用 IEEE sqrt + 除法, 不用 rsqrt: rsqrt 是 ~23 位近似, 会让 f32 path 失去 bit-exact
                // 一旦加 -ffast-math 或 -mrecip, 这里会被换成近似 rsqrt
                float inv_rms = 1.0f / std::sqrt(rms / hidden + eps);
                for (size_t j = 0; j < hidden; j++)
                {
                    result_data[i * hidden + j] = x_data[i * hidden + j] * inv_rms * weight_data[j];
                }
            }
            return result;
        }
        else if (x.dtype() == DType::BFloat16)
        {
            const bfloat16 *x_data = x.data<bfloat16>();
            const bfloat16 *weight_data = weight.data<bfloat16>();
            bfloat16 *result_data = result.data<bfloat16>();
            for (size_t i = 0; i < num_rows; i++)
            {
                float rms = 0;
                for (size_t j = 0; j < hidden; j++)
                {
                    float xij = static_cast<float>(x_data[i * hidden + j]);
                    rms += xij * xij;
                }
                // 故意用 IEEE sqrt + 除法, 不用 rsqrt: rsqrt 是 ~23 位近似, 会让 f32 path 失去 bit-exact
                // 一旦加 -ffast-math 或 -mrecip, 这里会被换成近似 rsqrt
                float inv_rms = 1.0f / std::sqrt(rms / hidden + eps);
                for (size_t j = 0; j < hidden; j++)
                {
                    // bfloat16 没有 operator*, 所以两个 bf16 隐式转 float、在 f32 下乘、结果 float 再 RNE 回 bf16
                    // 这恰好匹配 PyTorch CPU 上 bf16 乘法的语义(也是 upcast f32 算再 downcast)
                    result_data[i * hidden + j] = static_cast<bfloat16>(static_cast<float>(x_data[i * hidden + j]) * inv_rms) * weight_data[j];
                }
            }
            return result;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}
