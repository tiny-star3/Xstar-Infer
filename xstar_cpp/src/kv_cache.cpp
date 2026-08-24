#include <cstring>

#include "kv_cache.h"
#include "cuda/cuda_allocator.h"

KVCache::KVCache(std::int64_t num_layers, std::int64_t num_kv_heads, std::int64_t max_seq_len, std::int64_t head_dim, DType dtype, Device device) : num_layers_(num_layers), num_kv_heads_(num_kv_heads), max_seq_len_(max_seq_len), head_dim_(head_dim), dtype_(dtype), device_(device)
{
    cursor_ = 0;
    k_cache_.reserve(num_layers_);
    v_cache_.reserve(num_layers_);
    std::vector<std::int64_t> t_shape{num_kv_heads, max_seq_len, head_dim};
    for (std::int64_t i = 0; i < num_layers_; i++)
    {
        k_cache_.emplace_back(Tensor(t_shape, dtype_, device_));
        v_cache_.emplace_back(Tensor(t_shape, dtype_, device_));
    }
}

std::int64_t KVCache::cursor() const
{
    return cursor_;
}

void KVCache::write(std::int64_t layer, const Tensor &K, const Tensor &V, bool is_decode)
{
    if (layer < 0 || layer >= num_layers_)
        throw std::runtime_error("layer out-of-range");
    if (device_ != K.device() || K.device() != V.device())
        throw std::runtime_error("device mismatch");
    if (dtype_ != K.dtype() || K.dtype() != V.dtype())
        throw std::runtime_error("dtype mismatch");
    if (K.shape().size() != 3 || V.shape().size() != 3)
        throw std::runtime_error("rank mismatch");

    std::int64_t num_kv_heads = K.shape()[0];
    std::int64_t seq = K.shape()[1];
    std::int64_t head_dim = K.shape()[2];

    if (num_kv_heads != num_kv_heads_ || seq >= max_seq_len_ || seq < 0 || head_dim != head_dim_ || num_kv_heads != V.shape()[0] || seq != V.shape()[1] || head_dim != V.shape()[2])
        throw std::runtime_error("shape mismatch");

    if (device_ == Device::CUDA)
    {
        if (is_decode)
        {
            if (layer == 0)
            {
                if (cursor_ >= max_seq_len_)
                    throw std::runtime_error("kv cache full: cursor >= max_seq_len");
                cursor_++;
            }
            for (std::int64_t i = 0; i < num_kv_heads; i++)
            {
                cuda_memcpy_d2d(static_cast<void *>(static_cast<char *>(k_cache_[layer].data()) + (i * max_seq_len_ * head_dim + (cursor_ - 1) * head_dim) * dtype_size(K.dtype())), static_cast<const void *>(static_cast<const char *>(K.data()) + i * seq * head_dim * dtype_size(K.dtype())), head_dim * dtype_size(K.dtype()));
                cuda_memcpy_d2d(static_cast<void *>(static_cast<char *>(v_cache_[layer].data()) + (i * max_seq_len_ * head_dim + (cursor_ - 1) * head_dim) * dtype_size(V.dtype())), static_cast<const void *>(static_cast<const char *>(V.data()) + i * seq * head_dim * dtype_size(V.dtype())), head_dim * dtype_size(V.dtype()));
            }
        }
        else
        {
            cursor_ = seq;
            for (std::int64_t i = 0; i < num_kv_heads; i++)
            {
                cuda_memcpy_d2d(static_cast<void *>(static_cast<char *>(k_cache_[layer].data()) + i * max_seq_len_ * head_dim * dtype_size(K.dtype())), static_cast<const void *>(static_cast<const char *>(K.data()) + i * seq * head_dim * dtype_size(K.dtype())), seq * head_dim * dtype_size(K.dtype()));
                cuda_memcpy_d2d(static_cast<void *>(static_cast<char *>(v_cache_[layer].data()) + i * max_seq_len_ * head_dim * dtype_size(V.dtype())), static_cast<const void *>(static_cast<const char *>(V.data()) + i * seq * head_dim * dtype_size(V.dtype())), seq * head_dim * dtype_size(V.dtype()));
            }
        }
    }
    else if (device_ == Device::CPU)
    {
        if (is_decode)
        {
            if (layer == 0)
            {
                if (cursor_ >= max_seq_len_)
                    throw std::runtime_error("kv cache full: cursor >= max_seq_len");
                cursor_++;
            }
            for (std::int64_t i = 0; i < num_kv_heads; i++)
            {
                memcpy(static_cast<void *>(static_cast<char *>(k_cache_[layer].data()) + (i * max_seq_len_ * head_dim + (cursor_ - 1) * head_dim) * dtype_size(K.dtype())), static_cast<const void *>(static_cast<const char *>(K.data()) + i * seq * head_dim * dtype_size(K.dtype())), head_dim * dtype_size(K.dtype()));
                memcpy(static_cast<void *>(static_cast<char *>(v_cache_[layer].data()) + (i * max_seq_len_ * head_dim + (cursor_ - 1) * head_dim) * dtype_size(V.dtype())), static_cast<const void *>(static_cast<const char *>(V.data()) + i * seq * head_dim * dtype_size(V.dtype())), head_dim * dtype_size(V.dtype()));
            }
        }
        else
        {
            cursor_ = seq;
            for (std::int64_t i = 0; i < num_kv_heads; i++)
            {
                memcpy(static_cast<void *>(static_cast<char *>(k_cache_[layer].data()) + i * max_seq_len_ * head_dim * dtype_size(K.dtype())), static_cast<const void *>(static_cast<const char *>(K.data()) + i * seq * head_dim * dtype_size(K.dtype())), seq * head_dim * dtype_size(K.dtype()));
                memcpy(static_cast<void *>(static_cast<char *>(v_cache_[layer].data()) + i * max_seq_len_ * head_dim * dtype_size(V.dtype())), static_cast<const void *>(static_cast<const char *>(V.data()) + i * seq * head_dim * dtype_size(V.dtype())), seq * head_dim * dtype_size(V.dtype()));
            }
        }
    }
    else
        throw std::runtime_error("device mismatch");
}

Tensor KVCache::k_view(std::int64_t layer) const
{
    if (layer < 0 || layer >= num_layers_)
        throw std::runtime_error("layer out-of-range");

    Tensor k(std::vector<std::int64_t>{num_kv_heads_, cursor_, head_dim_}, dtype_, device_);
    if (device_ == Device::CUDA)
    {
        for (std::int64_t i = 0; i < num_kv_heads_; i++)
        {
            cuda_memcpy_d2d(static_cast<void *>(static_cast<char *>(k.data()) + i * cursor_ * head_dim_ * dtype_size(dtype_)), static_cast<const void *>(static_cast<const char *>(k_cache_[layer].data()) + i * max_seq_len_ * head_dim_ * dtype_size(dtype_)), cursor_ * head_dim_ * dtype_size(dtype_));
        }
    }
    else if (device_ == Device::CPU)
    {
        for (std::int64_t i = 0; i < num_kv_heads_; i++)
        {
            memcpy(static_cast<void *>(static_cast<char *>(k.data()) + i * cursor_ * head_dim_ * dtype_size(dtype_)), static_cast<const void *>(static_cast<const char *>(k_cache_[layer].data()) + i * max_seq_len_ * head_dim_ * dtype_size(dtype_)), cursor_ * head_dim_ * dtype_size(dtype_));
        }
    }
    else
        throw std::runtime_error("device mismatch");

    return k;
}

Tensor KVCache::v_view(std::int64_t layer) const
{
    if (layer < 0 || layer >= num_layers_)
        throw std::runtime_error("layer out-of-range");

    Tensor v(std::vector<std::int64_t>{num_kv_heads_, cursor_, head_dim_}, dtype_, device_);
    if (device_ == Device::CUDA)
    {
        for (std::int64_t i = 0; i < num_kv_heads_; i++)
        {
            cuda_memcpy_d2d(static_cast<void *>(static_cast<char *>(v.data()) + i * cursor_ * head_dim_ * dtype_size(dtype_)), static_cast<const void *>(static_cast<const char *>(v_cache_[layer].data()) + i * max_seq_len_ * head_dim_ * dtype_size(dtype_)), cursor_ * head_dim_ * dtype_size(dtype_));
        }
    }
    else if (device_ == Device::CPU)
    {
        for (std::int64_t i = 0; i < num_kv_heads_; i++)
        {
            memcpy(static_cast<void *>(static_cast<char *>(v.data()) + i * cursor_ * head_dim_ * dtype_size(dtype_)), static_cast<const void *>(static_cast<const char *>(v_cache_[layer].data()) + i * max_seq_len_ * head_dim_ * dtype_size(dtype_)), cursor_ * head_dim_ * dtype_size(dtype_));
        }
    }
    else
        throw std::runtime_error("device mismatch");

    return v;
}