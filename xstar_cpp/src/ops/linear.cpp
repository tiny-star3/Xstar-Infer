#include <stdexcept>
#include <vector>

#include "ops/linear.h"
#include "bfloat16.h"
#include "ops/gemm.h"
#include "ops/linear.h"

Tensor linear(const Tensor &x, const Tensor &weight, const Tensor *bias)
{
    if (x.dtype() != weight.dtype() || (bias != nullptr && bias->dtype() != x.dtype()))
        throw std::runtime_error("dtype mismatch");
    if (x.shape().size() < 1 || weight.shape().size() != 2)
        throw std::runtime_error("rank mismatch");
    if (weight.shape()[1] != x.shape().back() || (bias != nullptr && bias->shape()[0] != weight.shape()[0]))
        throw std::runtime_error("shape mismatch");
    if (x.device() != weight.device() || (bias != nullptr && weight.device() != bias->device()))
        throw std::runtime_error("device mismatch");

    std::vector<std::int64_t> result_shape(x.shape());
    result_shape[result_shape.size() - 1] = weight.shape()[0];
    Tensor result(result_shape, x.dtype(), x.device());
    std::int64_t k = x.shape().back();
    std::int64_t m = x.numel() / k;
    std::int64_t n = weight.shape()[0];
    if (x.device() == Device::CUDA)
    {
        gemm_launch(result.data(), x.data(), weight.data(), bias ? bias->data() : nullptr, m, k, n, true, k, k, n, x.dtype());

        return result;
    }
    else if (x.device() == Device::CPU)
    {
        if (x.dtype() == DType::Float32)
        {
            float *result_data = result.data<float>();
            const float *bias_data = bias ? bias->data<float>() : nullptr;
            gemm_cpu(x.data<float>(), weight.data<float>(), result_data, m, k, n, true, k, k, n);
            for (size_t i = 0; i < m; i++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    result_data[i * n + j] = result_data[i * n + j] + (bias_data ? bias_data[j] : 0.0f);
                }
            }
            return result;
        }
        else if (x.dtype() == DType::BFloat16)
        {
            bfloat16 *result_data = result.data<bfloat16>();
            const bfloat16 *bias_data = bias ? bias->data<bfloat16>() : nullptr;
            gemm_cpu(x.data<bfloat16>(), weight.data<bfloat16>(), result_data, m, k, n, true, k, k, n);
            for (size_t i = 0; i < m; i++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    result_data[i * n + j] = result_data[i * n + j] + (bias_data ? static_cast<float>(bias_data[j]) : 0.0f);
                }
            }
            return result;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}