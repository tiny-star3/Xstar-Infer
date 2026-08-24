#include <stdexcept>
#include <cmath>

#include "ops/mlp.h"
#include "ops/linear.h"
#include "bfloat16.h"

// sigmoid(x) = 1/(1+exp(-x))
// naive 形式里 exp(-x) 当 x < -88.7 时超过 f32 上界（e^88.7 ≈ 3.4e38 ≈ FLT_MAX，ln(FLT_MAX) ≈ 88.7），expf 返回 +inf
float sigmoid(float x)
{
    // 永远不会算 exp 的正参数
    if (x >= 0)
    {
        return 1.0f / (1.0f + expf(-x));
    }
    else
    {
        return expf(x) / (1.0f + expf(x));
    }
}

float silu(float x)
{
    return x * sigmoid(x);
}

template <typename T>
Tensor helper(const Tensor &gate_up, const Tensor &down_weight, std::int64_t num_rows, std::int64_t intermediate)
{
    Tensor act(std::vector<std::int64_t>{num_rows, intermediate}, gate_up.dtype(), gate_up.device());
    const T *gate_up_data = gate_up.data<T>();
    T *act_data = act.data<T>();
    for (size_t i = 0; i < num_rows; i++)
    {
        for (size_t j = 0; j < intermediate; j++)
        {
            float gate = static_cast<float>(gate_up_data[i * 2 * intermediate + j]);
            float up = static_cast<float>(gate_up_data[i * 2 * intermediate + j + intermediate]);
            // silu(gate)*up 整式 float、末尾 1 次 downcast, 比 PyTorch 少 1 次中间 downcast, 精度更高但不 bit-exact
            act_data[i * intermediate + j] = static_cast<T>(silu(gate) * up);
        }
    }
    return linear(act, down_weight, nullptr);
}

Tensor mlp(const Tensor &x, const Tensor &gate_up_weight, const Tensor &down_weight)
{
    if (x.dtype() != gate_up_weight.dtype() || gate_up_weight.dtype() != down_weight.dtype())
        throw std::runtime_error("dtype mismatch");
    if (x.shape().size() < 1 || gate_up_weight.shape().size() != 2 || down_weight.shape().size() != 2)
        throw std::runtime_error("rank mismatch");
    if (gate_up_weight.shape()[1] != x.shape()[x.shape().size() - 1] || down_weight.shape()[1] != gate_up_weight.shape()[0] / 2 || down_weight.shape()[0] != x.shape()[x.shape().size() - 1])
        throw std::runtime_error("shape mismatch");
    if (gate_up_weight.shape()[0] % 2 != 0)
        throw std::runtime_error("gate_up out must be even");
    if (x.device() != gate_up_weight.device() || gate_up_weight.device() != down_weight.device())
        throw std::runtime_error("device mismatch");

    std::int64_t hidden = gate_up_weight.shape()[1];
    std::int64_t num_rows = x.numel() / hidden;
    std::int64_t intermediate = down_weight.shape()[1];

    if (x.device() == Device::CUDA)
    {
        Tensor act(std::vector<std::int64_t>{num_rows, intermediate}, x.dtype(), x.device());
        gemm_silu_and_mul_launch(act.data(), x.data(), gate_up_weight.data(), num_rows, hidden, intermediate, intermediate, x.dtype());

        return linear(act, down_weight, nullptr);
    }
    else if (x.device() == Device::CPU)
    {
        Tensor gate_up = linear(x, gate_up_weight, nullptr);
        if (x.dtype() == DType::Float32)
        {
            return helper<float>(gate_up, down_weight, num_rows, intermediate);
        }
        else if (x.dtype() == DType::BFloat16)
        {
            return helper<bfloat16>(gate_up, down_weight, num_rows, intermediate);
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}