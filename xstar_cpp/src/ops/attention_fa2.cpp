#include <stdexcept>

#include "ops/attention_fa2.h"

Tensor attention_fa2(const Tensor &Q, const Tensor &K, const Tensor &V, const Tensor *mask)
{
    if (Q.shape().size() != 3 || K.shape().size() != 3 || V.shape().size() != 3 || (mask != nullptr && mask->shape().size() != 2))
        throw std::runtime_error("rank mismatch");
    if (Q.dtype() != K.dtype() || K.dtype() != V.dtype() || (mask != nullptr && V.dtype() != mask->dtype()))
        throw std::runtime_error("dtype mismatch");
    if ((Q.shape()[1] != 1 && Q.shape()[1] != K.shape()[1]) || K.shape()[1] != V.shape()[1] || K.shape()[0] != V.shape()[0] || Q.shape()[2] != K.shape()[2] || K.shape()[2] != V.shape()[2] || (mask != nullptr && (V.shape()[1] != mask->shape()[0] || mask->shape()[0] != mask->shape()[1])))
        throw std::runtime_error("shape mismatch");
    if (Q.device() != K.device() || K.device() != V.device() || (mask && V.device() != mask->device()))
        throw std::runtime_error("device mismatch");

    std::int64_t num_heads = Q.shape()[0];
    std::int64_t seq_q = Q.shape()[1];
    std::int64_t seq_k = K.shape()[1];
    std::int64_t head_dim = Q.shape()[2];
    std::int64_t num_key_value_heads = K.shape()[0];

    if (num_heads % num_key_value_heads != 0)
        throw std::runtime_error("rep not integral");

    std::int64_t rep = num_heads / num_key_value_heads;
    std::int64_t batch = 1;
    Tensor result(std::vector<std::int64_t>{seq_q, num_heads * head_dim}, Q.dtype(), Q.device());

    if (Q.device() == Device::CUDA)
    {
        if (seq_q == 1)
        {
            flash_attention2_decode_launch(result.data(), Q.data(), K.data(), V.data(), batch, num_heads, num_key_value_heads, seq_k, head_dim, Q.dtype());
            return result;
        }
        else
        {
            flash_attention2_prefill_launch(result.data(), Q.data(), K.data(), V.data(), mask ? mask->data() : mask, batch, num_heads, num_key_value_heads, seq_q, seq_k, head_dim, Q.dtype());
            return result;
        }
    }
    else
        throw std::runtime_error("unsupported device");
}