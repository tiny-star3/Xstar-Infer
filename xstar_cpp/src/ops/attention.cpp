#include <stdexcept>
#include <limits>
#include <vector>
#include <cmath>

#include "ops/attention.h"
#include "ops/softmax.h"
#include "bfloat16.h"
#include "ops/gemm.h"

Tensor attention(const Tensor &Q, const Tensor &K, const Tensor &V, const Tensor *mask)
{
    if (Q.shape().size() != 3 || K.shape().size() != 3 || V.shape().size() != 3 || (mask != nullptr && mask->shape().size() != 2))
        throw std::runtime_error("rank mismatch");
    if (Q.dtype() != K.dtype() || K.dtype() != V.dtype() || (mask != nullptr && V.dtype() != mask->dtype()))
        throw std::runtime_error("dtype mismatch");
    if (Q.shape()[1] != K.shape()[1] || K.shape()[1] != V.shape()[1] || K.shape()[0] != V.shape()[0] || Q.shape()[2] != K.shape()[2] || K.shape()[2] != V.shape()[2] || (mask != nullptr && (V.shape()[1] != mask->shape()[0] || mask->shape()[0] != mask->shape()[1])))
        throw std::runtime_error("shape mismatch");
    if (Q.device() != K.device() || K.device() != V.device() || (mask && V.device() != mask->device()))
        throw std::runtime_error("device mismatch");

    std::int64_t num_heads = Q.shape()[0];
    std::int64_t seq = Q.shape()[1];
    std::int64_t head_dim = Q.shape()[2];
    std::int64_t num_key_value_heads = K.shape()[0];

    if (num_heads % num_key_value_heads != 0)
        throw std::runtime_error("rep not integral");

    std::int64_t rep = num_heads / num_key_value_heads;
    Tensor qk(std::vector<std::int64_t>{num_heads, seq, seq}, Q.dtype(), Q.device());
    Tensor result(std::vector<std::int64_t>{seq, num_heads * head_dim}, Q.dtype(), Q.device());
    float scalar = 1.0 / sqrt(head_dim);
    if (Q.device() == Device::CUDA)
    {
        for (std::int64_t h = 0; h < num_heads; h++)
        {
            gemm_launch(static_cast<void *>(static_cast<char *>(qk.data()) + h * seq * seq * dtype_size(qk.dtype())), static_cast<const void *>(static_cast<const char *>(Q.data()) + h * seq * head_dim * dtype_size(Q.dtype())), static_cast<const void *>(static_cast<const char *>(K.data()) + (h / rep) * seq * head_dim * dtype_size(K.dtype())), nullptr, seq, head_dim, seq, true, head_dim, head_dim, seq, Q.dtype());
        }
        scale_mask_launch(qk.data(), mask ? mask->data() : mask, scalar, num_heads, seq, qk.dtype());
        Tensor attn_weights = softmax(qk, -1);
        for (std::int64_t h = 0; h < num_heads; h++)
        {
            gemm_launch(static_cast<void *>(static_cast<char *>(result.data()) + h * head_dim * dtype_size(result.dtype())), static_cast<void *>(static_cast<char *>(attn_weights.data()) + h * seq * seq * dtype_size(attn_weights.dtype())), static_cast<const void *>(static_cast<const char *>(V.data()) + (h / rep) * seq * head_dim * dtype_size(V.dtype())), nullptr, seq, seq, head_dim, false, seq, head_dim, num_heads * head_dim, attn_weights.dtype());
        }
        return result;
    }
    else if (Q.device() == Device::CPU)
    {
        if (Q.dtype() == DType::Float32)
        {
            const float *Q_data = Q.data<float>();
            const float *K_data = K.data<float>();
            const float *V_data = V.data<float>();
            float *qk_data = qk.data<float>();
            float *result_data = result.data<float>();
            const float *mask_data = mask ? mask->data<float>() : nullptr;
            for (std::int64_t h = 0; h < num_heads; h++)
            {
                gemm_cpu<float>(Q_data + h * seq * head_dim, K_data + (h / rep) * seq * head_dim, qk_data + h * seq * seq, seq, head_dim, seq, true, head_dim, head_dim, seq);
                for (std::int64_t i = 0; i < seq; i++)
                {
                    for (std::int64_t j = 0; j < seq; j++)
                    {
                        if (mask_data != nullptr)
                        {
                            qk_data[h * seq * seq + i * seq + j] = qk_data[h * seq * seq + i * seq + j] * scalar;
                            qk_data[h * seq * seq + i * seq + j] += mask_data[i * seq + j];
                        }
                        else if (i < j)
                        {
                            qk_data[h * seq * seq + i * seq + j] = -std::numeric_limits<float>::infinity();
                        }
                        else
                        {
                            qk_data[h * seq * seq + i * seq + j] = qk_data[h * seq * seq + i * seq + j] * scalar;
                        }
                    }
                }
            }
            Tensor attn_weights = softmax(qk, -1);
            float *attn_weights_data = attn_weights.data<float>();
            for (size_t h = 0; h < num_heads; h++)
            {
                gemm_cpu<float>(attn_weights_data + h * seq * seq, V_data + (h / rep) * seq * head_dim, result_data + h * head_dim, seq, seq, head_dim, false, seq, head_dim, num_heads * head_dim);
            }
            return result;
        }
        else if (Q.dtype() == DType::BFloat16)
        {
            const bfloat16 *Q_data = Q.data<bfloat16>();
            const bfloat16 *K_data = K.data<bfloat16>();
            const bfloat16 *V_data = V.data<bfloat16>();
            bfloat16 *qk_data = qk.data<bfloat16>();
            bfloat16 *result_data = result.data<bfloat16>();
            const bfloat16 *mask_data = mask ? mask->data<bfloat16>() : nullptr;
            for (size_t h = 0; h < num_heads; h++)
            {
                gemm_cpu<bfloat16>(Q_data + h * seq * head_dim, K_data + (h / rep) * seq * head_dim, qk_data + h * seq * seq, seq, head_dim, seq, true, head_dim, head_dim, seq);
                for (size_t i = 0; i < seq; i++)
                {
                    for (size_t j = 0; j < seq; j++)
                    {
                        // bfloat16 没有 operator*, 所以两个 bf16 隐式转 float、在 f32 下乘、结果 float 再 RNE 回 bf16
                        // 这恰好匹配 PyTorch CPU 上 bf16 乘法的语义(也是 upcast f32 算再 downcast)
                        qk_data[h * seq * seq + i * seq + j] = qk_data[h * seq * seq + i * seq + j] * scalar;
                        if (mask_data != nullptr)
                        {
                            qk_data[h * seq * seq + i * seq + j] = qk_data[h * seq * seq + i * seq + j] + mask_data[i * seq + j];
                        }
                        else if (mask_data == nullptr && i < j)
                        {
                            qk_data[h * seq * seq + i * seq + j] = static_cast<bfloat16>(-std::numeric_limits<float>::infinity());
                        }
                    }
                }
            }
            Tensor attn_weights = softmax(qk, -1);
            bfloat16 *attn_weights_data = attn_weights.data<bfloat16>();
            for (size_t h = 0; h < num_heads; h++)
            {
                gemm_cpu<bfloat16>(attn_weights_data + h * seq * seq, V_data + (h / rep) * seq * head_dim, result_data + h * head_dim, seq, seq, head_dim, false, seq, head_dim, num_heads * head_dim);
            }
            return result;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}