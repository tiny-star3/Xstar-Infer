#include <stdexcept>
#include <string>

#include "block_manager.h"
#include "cuda/cuda_allocator.h"

BlockManager::BlockManager(int num_blocks, int block_size, int kv_slot_bytes, Device dev, int num_layers) : num_blocks_(num_blocks), block_size_(block_size), block_bytes_(block_size * kv_slot_bytes), num_layers_(num_layers), layer_stride_(num_blocks * block_bytes_), dev_(dev), num_allocated_(0)
{
    // 0 合法, 空池
    if (num_blocks < 0)
        throw std::runtime_error("num_blocks must be non-negative");
    // 防止 block_bytes_=0, slot 算术崩
    if (block_size <= 0 || kv_slot_bytes <= 0)
        throw std::runtime_error("block_size and kv_slot_bytes must be positive");
    // GPU-only, CPU 留给以后
    if (dev != Device::CUDA)
        throw std::runtime_error("BlockManager currently supports CUDA only");
    if (num_layers <= 0)
        throw std::runtime_error("num_layers must be greater than 0");

    // blocks_ 构造在 kv_pool_ 之前, 防止 resize 抛 bad_alloc， kv_pool_ 已分配但析构不会跑(对象没构造完) → 泄漏
    blocks_.resize(num_blocks_);
    kv_pool_ = cuda_alloc(static_cast<std::size_t>(num_layers) * num_blocks_ * block_bytes_);
    for (int i = 0; i < num_blocks_; i++)
    {
        blocks_[i].block_id = i;
        blocks_[i].ref_cnt = 0;
        blocks_[i].prev_free = (i == 0) ? &head_sentinel_ : &blocks_[i - 1];
        blocks_[i].next_free = (i == num_blocks_ - 1) ? &tail_sentinel_ : &blocks_[i + 1];
    }
    // 哨兵两向链接(num_blocks_==0 时两哨兵互指, 空链表正确)
    head_sentinel_.next_free = (num_blocks_ == 0) ? &tail_sentinel_ : &blocks_[0];
    tail_sentinel_.prev_free = (num_blocks_ == 0) ? &head_sentinel_ : &blocks_[num_blocks_ - 1];
    head_sentinel_.prev_free = nullptr;
    tail_sentinel_.next_free = nullptr;
    head_sentinel_.block_id = -1;
    tail_sentinel_.block_id = -1;
    head_sentinel_.ref_cnt = 0;
    tail_sentinel_.ref_cnt = 0;
}

BlockManager::~BlockManager()
{
    cuda_free(kv_pool_);
}

std::vector<int> BlockManager::alloc(int num)
{
    // num==0 返回空 vector
    if (num < 0)
        throw std::runtime_error("alloc count must be non-negative");
    if (num_free() < num)
        throw std::runtime_error("insufficient free blocks");

    std::vector<int> result;
    while (num--)
    {
        int block_id = pop_free_head();
        result.push_back(block_id);
    }

    return result;
}

void BlockManager::free(const std::vector<int> &block_ids)
{
    for (auto block_id : block_ids)
    {
        check_id(block_id);
        check_allocated(block_id);
        if (--blocks_[block_id].ref_cnt == 0)
        {
            push_free_head(&blocks_[block_id]);
        }
    }
}

void BlockManager::ref(const std::vector<int> &block_ids)
{
    for (auto block_id : block_ids)
    {
        check_id(block_id);
        check_allocated(block_id);
        blocks_[block_id].ref_cnt++;
    }
}

std::vector<int> BlockManager::fork(const std::vector<int> &src_table)
{
    ref(src_table);

    return src_table;
}

int BlockManager::write_block(int block_id)
{
    check_id(block_id);
    check_allocated(block_id);
    if (blocks_[block_id].ref_cnt == 1)
    {
        return block_id;
    }
    else
    {
        if (num_free() < 1)
            throw std::runtime_error("insufficient free blocks");
        int new_block_id = pop_free_head();
        blocks_[block_id].ref_cnt--;
        cow_copy(block_id, new_block_id);

        return new_block_id;
    }
}

void BlockManager::cow_copy(int src_id, int dst_id)
{
    check_id(src_id);
    check_id(dst_id);
    for (int i = 0; i < num_layers_; i++)
    {
        char *base = static_cast<char *>(layer_base(i));
        cuda_memcpy_d2d(static_cast<void *>(base + static_cast<std::size_t>(dst_id) * block_bytes_),
                        static_cast<void *>(base + static_cast<std::size_t>(src_id) * block_bytes_),
                        static_cast<std::size_t>(block_bytes_));
    }
}

int BlockManager::num_free() const
{
    return num_blocks_ - num_allocated_;
}

int BlockManager::num_allocated() const
{
    return num_allocated_;
}

int BlockManager::block_ref_cnt(int block_id) const
{
    return blocks_[block_id].ref_cnt;
}

void *BlockManager::pool_ptr() const
{
    return kv_pool_;
}

int64_t BlockManager::layer_stride() const
{
    return layer_stride_;
}

int BlockManager::block_bytes() const
{
    return block_bytes_;
}

int BlockManager::block_size() const
{
    return block_size_;
}

void *BlockManager::layer_base(int layer) const
{
    if (layer < 0 || layer >= num_layers_)
        throw std::runtime_error("layer out-of-range");
    return static_cast<void *>(static_cast<char *>(kv_pool_) + layer * layer_stride_);
}

int BlockManager::num_layers() const
{
    return num_layers_;
}

void BlockManager::check_id(int block_id) const
{
    if (block_id < 0 || block_id >= num_blocks_)
        throw std::runtime_error(std::string("block_id out of range: ") + std::to_string(block_id));
}
void BlockManager::check_allocated(int block_id) const
{
    if (blocks_[block_id].ref_cnt < 1)
        throw std::runtime_error(std::string("block not allocated: ") + std::to_string(block_id));
}

int BlockManager::pop_free_head()
{
    KVCacheBlock *result = head_sentinel_.next_free;
    // 哨兵封尾, 判空
    if (head_sentinel_.next_free == &tail_sentinel_)
    {
        throw std::runtime_error("pop_free_head: free list empty");
    }
    result->next_free->prev_free = &head_sentinel_;
    head_sentinel_.next_free = result->next_free;
    result->prev_free = nullptr;
    result->next_free = nullptr;
    result->ref_cnt = 1;
    num_allocated_++;

    return result->block_id;
}

void BlockManager::push_free_head(KVCacheBlock *block)
{
    num_allocated_--;
    head_sentinel_.next_free->prev_free = block;
    block->next_free = head_sentinel_.next_free;
    block->prev_free = &head_sentinel_;
    head_sentinel_.next_free = block;
    block->ref_cnt = 0;
}