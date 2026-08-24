#include "ops/add.h"
#include "bfloat16.h"

Tensor add(const Tensor &a, const Tensor &b)
{
    if (a.shape().size() != b.shape().size())
        throw std::runtime_error("add shape mismatch");
    for (size_t i = 0; i < a.shape().size(); i++)
        if (a.shape()[i] != b.shape()[i])
            throw std::runtime_error("add shape mismatch");
    if (a.dtype() != b.dtype())
        throw std::runtime_error("add dtype mismatch");
    if (a.device() != b.device())
        throw std::runtime_error("add device mismatch");

    size_t numel = a.numel();
    Tensor result(a.shape(), a.dtype(), a.device());

    if (a.device() == Device::CUDA)
    {
        add_launch(result.data(), a.data(), b.data(), numel, a.dtype());

        return result;
    }
    else if (a.device() == Device::CPU)
    {
        if (a.dtype() == DType::Float32)
        {
            const float *a_data = a.data<float>();
            const float *b_data = b.data<float>();
            float *result_data = result.data<float>();
            for (size_t i = 0; i < numel; i++)
            {
                result_data[i] = a_data[i] + b_data[i];
            }
            return result;
        }
        else if (a.dtype() == DType::BFloat16)
        {
            const bfloat16 *a_data = a.data<bfloat16>();
            const bfloat16 *b_data = b.data<bfloat16>();
            bfloat16 *result_data = result.data<bfloat16>();
            for (size_t i = 0; i < numel; i++)
            {
                // bfloat16 没有 operator+, 所以两个 bf16 隐式转 float、在 f32 下加、结果 float 再 RNE 回 bf16
                result_data[i] = a_data[i] + b_data[i];
            }
            return result;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}