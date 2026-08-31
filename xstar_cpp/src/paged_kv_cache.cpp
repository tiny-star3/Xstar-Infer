#include <stdexcept>
#include <memory>

#include "paged_kv_cache.h"
#include "cuda/cuda_allocator.h"
#include "ops/paged_write.h"

PagedKVCache::PagedKVCache(std::int64_t num_kv_heads, std::int64_t head_dim, std::int64_t max_seq_len, std::int64_t block_size, DType dtype, Device device) : num_kv_heads_(num_kv_heads), head_dim_(head_dim), max_seq_len_(max_seq_len), block_size_(block_size), dtype_(dtype), device_(device), d_block_table_(nullptr), adopted_(false)
{
    cursor_ = 0;
    d_block_table_cap_ = 0;
    block_size_checked_ = false;
}

PagedKVCache::~PagedKVCache()
{
    cuda_free(d_block_table_);
}

std::int64_t PagedKVCache::cursor() const
{
    return cursor_;
}

const std::vector<int> &PagedKVCache::block_table() const
{
    return block_table_;
}

const int *PagedKVCache::d_block_table() const
{
    return d_block_table_;
}

int PagedKVCache::block_size() const
{
    return block_size_;
}

void PagedKVCache::write(std::int64_t layer, BlockManager &bm, const Tensor &K, const Tensor &V, bool is_decode)
{
    if (!block_size_checked_)
    {
        if (bm.block_size() != block_size_)
            throw std::runtime_error("bm.block_size() != block_size_");
        else
            block_size_checked_ = true;
    }
    if (layer < 0 || layer >= bm.num_layers())
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

    if (num_kv_heads != num_kv_heads_ || cursor_ + seq >= max_seq_len_ || seq < 0 || head_dim != head_dim_ || num_kv_heads != V.shape()[0] || seq != V.shape()[1] || head_dim != V.shape()[2])
        throw std::runtime_error("shape mismatch");

    if (device_ == Device::CUDA)
    {
        if (is_decode)
        {
            if (d_block_table_ == nullptr)
                throw std::runtime_error("prefill precedes decode");
            if (layer == 0)
            {
                if (cursor_ >= max_seq_len_)
                    throw std::runtime_error("kv cache full: cursor >= max_seq_len");
                cursor_++;
                if (cursor_ % block_size_ == 1)
                {
                    std::vector<int> new_blocks = bm.alloc(1);
                    cuda_memcpy_h2d(d_block_table_ + block_table_.size(), new_blocks.data(), sizeof(int));
                    block_table_.push_back(new_blocks[0]);
                }
            }
            int slot_mapping = block_table_[(cursor_ - 1) / block_size_] * block_size_ + (cursor_ - 1) % block_size_;
            paged_write(bm, layer, K, V, &slot_mapping);
        }
        else
        {
            if (layer == 0)
            {
                if (cursor_ > 0 && !adopted_)
                    throw std::runtime_error("second prefill; adopt_prefix() first");

                if (cursor_ > 0)
                {
                    // 拥有 prefix 的 prefill
                    std::vector<int> prefill_block_table = bm.alloc((seq + block_size_ - 1) / block_size_);
                    int bt_size = block_table_.size();
                    block_table_.insert(block_table_.end(), prefill_block_table.begin(), prefill_block_table.end());
                    cursor_ += seq;
                    cuda_memcpy_h2d(d_block_table_ + bt_size, prefill_block_table.data(), prefill_block_table.size() * sizeof(int));
                }
                else
                {
                    // 普通 prefill
                    block_table_ = std::move(bm.alloc((seq + block_size_ - 1) / block_size_));
                    cursor_ = seq;
                    d_block_table_cap_ = (max_seq_len_ + block_size_ - 1) / block_size_;
                    d_block_table_ = static_cast<int *>(cuda_alloc(d_block_table_cap_ * sizeof(int)));
                    cuda_memcpy_h2d(d_block_table_, block_table_.data(), block_table_.size() * sizeof(int));
                }
            }
            std::unique_ptr<int[]> slot_mapping(new int[seq]);
            int cursor_before = cursor_ - seq;
            for (std::int64_t i = 0; i < seq; i++)
            {
                slot_mapping[i] = block_table_[(cursor_before + i) / block_size_] * block_size_ + (cursor_before + i) % block_size_;
            }
            paged_write(bm, layer, K, V, slot_mapping.get());
        }
    }
    else
        throw std::runtime_error("device mismatch");
}

void PagedKVCache::prepare_meta(std::int64_t layer, BlockManager &bm, std::int64_t len, bool is_decode)
{
    if (!block_size_checked_)
    {
        if (bm.block_size() != block_size_)
            throw std::runtime_error("bm.block_size() != block_size_");
        else
            block_size_checked_ = true;
    }
    if (layer < 0 || layer >= bm.num_layers())
        throw std::runtime_error("layer out-of-range");
    // meta 只 layer 0 做, 别的层直接返回
    if (layer != 0)
        return;

    if (len >= max_seq_len_ || len < 0)
        throw std::runtime_error("len >= max_seq_len_ or len < 0");

    if (device_ == Device::CUDA)
    {
        if (is_decode)
        {
            if (d_block_table_ == nullptr)
                throw std::runtime_error("prefill precedes decode");
            if (len != 1)
                throw std::runtime_error("decode, but len != 1");
            if (cursor_ >= max_seq_len_)
                throw std::runtime_error("kv cache full: cursor >= max_seq_len");
            cursor_++;
            if (cursor_ % block_size_ == 1)
            {
                std::vector<int> new_blocks = bm.alloc(1);
                cuda_memcpy_h2d(d_block_table_ + block_table_.size(), new_blocks.data(), sizeof(int));
                block_table_.push_back(new_blocks[0]);
            }
        }
        else
        {
            if (cursor_ > 0 && !adopted_)
                throw std::runtime_error("second prefill; adopt_prefix() first");

            if (cursor_ > 0)
            {
                // 拥有 prefix 的 prefill
                std::vector<int> prefill_block_table = bm.alloc((len + block_size_ - 1) / block_size_);
                int bt_size = block_table_.size();
                block_table_.insert(block_table_.end(), prefill_block_table.begin(), prefill_block_table.end());
                cursor_ += len;
                cuda_memcpy_h2d(d_block_table_ + bt_size, prefill_block_table.data(), prefill_block_table.size() * sizeof(int));
            }
            else
            {
                // 普通 prefill
                block_table_ = std::move(bm.alloc((len + block_size_ - 1) / block_size_));
                cursor_ = len;
                d_block_table_cap_ = (max_seq_len_ + block_size_ - 1) / block_size_;
                d_block_table_ = static_cast<int *>(cuda_alloc(d_block_table_cap_ * sizeof(int)));
                cuda_memcpy_h2d(d_block_table_, block_table_.data(), block_table_.size() * sizeof(int));
            }
        }
    }
    else
        throw std::runtime_error("device mismatch");
}

void PagedKVCache::reset()
{
    adopted_ = false;
    cursor_ = 0;
    block_table_.clear();
    cuda_free(d_block_table_);
    d_block_table_ = nullptr;
    d_block_table_cap_ = 0;
}

void PagedKVCache::adopt_prefix(const std::vector<int> &prefix_blocks)
{
    if (cursor_ != 0 || !block_table_.empty() || d_block_table_)
    {
        throw std::runtime_error("cache is not fresh");
    }
    if (prefix_blocks.empty())
    {
        throw std::runtime_error("empty prefix; adopt does nothing");
    }

    std::int64_t matched_len = prefix_blocks.size() * block_size_;

    if (matched_len >= max_seq_len_)
    {
        throw std::runtime_error("matched_len >= max_seq_len");
    }

    adopted_ = true;
    cursor_ = matched_len;
    block_table_ = prefix_blocks;
    d_block_table_cap_ = (max_seq_len_ + block_size_ - 1) / block_size_;
    d_block_table_ = static_cast<int *>(cuda_alloc(d_block_table_cap_ * sizeof(int)));
    cuda_memcpy_h2d(d_block_table_, block_table_.data(), block_table_.size() * sizeof(int));
}
