#include <stdexcept>
#include <algorithm>
#include <functional>

#include "radix_cache.h"

LRUList::LRUList()
{
    head_sentinel_.lru_next = &tail_sentinel_;
    tail_sentinel_.lru_prev = &head_sentinel_;
}

void LRUList::push_back(RadixNode *node)
{
    if (node->in_lru)
        throw std::runtime_error("LRUList::push_back: node already in LRUList");

    node->in_lru = true;
    tail_sentinel_.lru_prev->lru_next = node;
    node->lru_prev = tail_sentinel_.lru_prev;
    node->lru_next = &tail_sentinel_;
    tail_sentinel_.lru_prev = node;
    size_++;
}

RadixNode *LRUList::pop_front()
{
    if (empty())
        throw std::runtime_error("LRUList::pop_front: LRUList is empty");

    RadixNode *result = head_sentinel_.lru_next;
    head_sentinel_.lru_next = result->lru_next;
    result->lru_next->lru_prev = &head_sentinel_;
    result->lru_next = nullptr;
    result->lru_prev = nullptr;
    result->in_lru = false;
    size_--;

    return result;
}

void LRUList::move_to_back(RadixNode *node)
{
    if (!node->in_lru)
        throw std::runtime_error("LRUList::move_to_back: node not in LRUList");

    node->lru_prev->lru_next = node->lru_next;
    node->lru_next->lru_prev = node->lru_prev;
    tail_sentinel_.lru_prev->lru_next = node;
    node->lru_prev = tail_sentinel_.lru_prev;
    node->lru_next = &tail_sentinel_;
    tail_sentinel_.lru_prev = node;
}

void LRUList::remove(RadixNode *node)
{
    if (!node->in_lru)
        throw std::runtime_error("LRUList::remove: node not in LRUList");

    node->lru_prev->lru_next = node->lru_next;
    node->lru_next->lru_prev = node->lru_prev;
    node->lru_next = nullptr;
    node->lru_prev = nullptr;
    node->in_lru = false;
    size_--;
}

bool LRUList::empty() const
{
    return head_sentinel_.lru_next == &tail_sentinel_;
}

int LRUList::size() const
{
    return size_;
}

RadixTree::RadixTree(int block_size)
{
    block_size_ = block_size;
    root_ = new RadixNode();
    root_->lock_ref = 1;
    evictable_blocks_ = 0;
}

RadixTree::~RadixTree()
{
    std::function<void(RadixNode *)> DeleteTree = [&](RadixNode *now)
    {
        for (auto &node : now->children)
        {
            DeleteTree(node.second);
        }
        delete now;
    };
    DeleteTree(root_);
}

std::pair<int, RadixNode *> RadixTree::match_prefix(const std::vector<int> &tokens)
{
    // 输入截断, 残余不参与 match
    int total = (tokens.size() / block_size_) * block_size_;
    int matched_length = 0;
    RadixNode *now = root_;
    while (now)
    {
        if (now->in_lru)
        {
            // 命中刷新
            lru_.move_to_back(now);
        }
        int m_key;
        for (m_key = 0; m_key < now->key.size() && matched_length + m_key < total; m_key++)
        {
            if (now->key[m_key] != tokens[matched_length + m_key])
            {
                break;
            }
        }
        matched_length += m_key;
        // 内部分叉
        if (m_key < now->key.size())
        {
            return std::make_pair(matched_length / block_size_ * block_size_, now);
        }
        // tokens 比完
        if (matched_length >= total)
        {
            return std::make_pair(matched_length / block_size_ * block_size_, now);
        }
        // 余下不足一块, 无法查叉
        if (total - matched_length < block_size_)
        {
            return std::make_pair(matched_length / block_size_ * block_size_, now);
        }
        std::vector<int> first_block(tokens.begin() + matched_length, tokens.begin() + matched_length + block_size_);
        auto it = now->children.find(first_block);
        // 没这叉, 停
        if (it == now->children.end())
        {
            return std::make_pair(matched_length / block_size_ * block_size_, now);
        }
        now = it->second;
    }
    return {matched_length / block_size_ * block_size_, root_};
}

RadixNode *RadixTree::insert(const std::vector<int> &tokens, const std::vector<int> &block_table)
{
    // 输入截断, 残余不参与插入
    int total = (tokens.size() / block_size_) * block_size_;
    int matched_length = 0;
    RadixNode *now = root_;
    while (now)
    {
        int m_key;
        for (m_key = 0; m_key < now->key.size() && matched_length + m_key < total; m_key++)
        {
            if (now->key[m_key] != tokens[matched_length + m_key])
            {
                break;
            }
        }
        matched_length += m_key;
        // 内部分叉
        if (m_key < now->key.size())
        {
            now = _split_node(now, m_key / block_size_ * block_size_);
            // tokens 是 node key 的真前缀
            if (matched_length == total)
            {
                return now;
            }
            RadixNode *leaf = new RadixNode();
            leaf->parent = now;
            leaf->key = std::vector<int>(tokens.begin() + matched_length, tokens.begin() + total);
            leaf->block_table = std::vector<int>(block_table.begin() + matched_length / block_size_, block_table.begin() + total / block_size_);
            lru_.push_back(leaf);
            evictable_blocks_ += leaf->block_table.size();
            now->children[std::vector<int>(leaf->key.begin(), leaf->key.begin() + block_size_)] = leaf;
            return leaf;
        }
        // tokens 比完
        if (matched_length == total)
        {
            return now;
        }
        std::vector<int> first_block(tokens.begin() + matched_length, tokens.begin() + matched_length + block_size_);
        auto it = now->children.find(first_block);
        // 没这叉, 加
        if (it == now->children.end())
        {
            RadixNode *leaf = new RadixNode();
            leaf->parent = now;
            leaf->key = std::vector<int>(tokens.begin() + matched_length, tokens.begin() + total);
            leaf->block_table = std::vector<int>(block_table.begin() + matched_length / block_size_, block_table.begin() + total / block_size_);
            lru_.push_back(leaf);
            evictable_blocks_ += leaf->block_table.size();
            now->children[std::vector<int>(leaf->key.begin(), leaf->key.begin() + block_size_)] = leaf;
            return leaf;
        }
        now = it->second;
    }

    return now;
}

void RadixTree::inc_lock_ref(RadixNode *node)
{
    while (node)
    {
        if (node->lock_ref++ == 0 && node->in_lru)
        {
            evictable_blocks_ -= node->block_table.size();
            lru_.remove(node);
        }
        node = node->parent;
    }
}

void RadixTree::dec_lock_ref(RadixNode *node)
{
    while (node)
    {
        if (--node->lock_ref == 0 && !node->in_lru && node->children.empty())
        {
            lru_.push_back(node);
            evictable_blocks_ += node->block_table.size();
        }
        node = node->parent;
    }
}

int RadixTree::evict(int need_blocks, BlockManager &bm)
{
    int freed_blocks = 0;
    while (freed_blocks < need_blocks && !lru_.empty())
    {
        RadixNode *node = lru_.pop_front();
        RadixNode *parent = node->parent;
        evictable_blocks_ -= node->block_table.size();
        freed_blocks += node->block_table.size();
        _delete_leaf(node, bm);
        if (parent != root_ && parent->children.empty() && parent->lock_ref == 0 && !parent->in_lru)
        {
            lru_.push_back(parent);
            evictable_blocks_ += parent->block_table.size();
        }
    }
    return freed_blocks;
}

int RadixTree::lru_size() const
{
    return lru_.size();
}

int RadixTree::evictable_blocks() const
{
    return evictable_blocks_;
}

RadixNode *RadixTree::_split_node(RadixNode *child, int split_len)
{
    if (split_len <= 0 || split_len % block_size_ != 0 || (int)child->key.size() - split_len < block_size_)
        throw std::runtime_error("RadixTree::_split_node: split_len must be positive, block-aligned, and leave child->key >= block_size");

    std::vector<int> node_key(child->key.begin(), child->key.begin() + split_len);
    child->key = std::vector<int>(child->key.begin() + split_len, child->key.end());
    std::vector<int> node_block_table(child->block_table.begin(), child->block_table.begin() + split_len / block_size_);
    child->block_table = std::vector<int>(child->block_table.begin() + split_len / block_size_, child->block_table.end());

    std::vector<int> node_first_block(node_key.begin(), node_key.begin() + block_size_);
    std::vector<int> child_first_block(child->key.begin(), child->key.begin() + block_size_);
    RadixNode *node = new RadixNode();
    node->children[child_first_block] = child;
    node->parent = child->parent;
    node->key = node_key;
    node->block_table = node_block_table;
    // split preserves pin: a request pinning child's prefix also pins new_node (its prefix)
    node->lock_ref = child->lock_ref;
    child->parent->children[node_first_block] = node;
    child->parent = node;
    if (child->in_lru)
    {
        evictable_blocks_ -= node->block_table.size();
    }

    return node;
}

void RadixTree::_delete_leaf(RadixNode *node, BlockManager &bm)
{
    std::vector<int> first_block(node->key.begin(), node->key.begin() + block_size_);
    node->parent->children.erase(first_block);
    if (node->in_lru)
    {
        evictable_blocks_ -= node->block_table.size();
        lru_.remove(node);
    }
    bm.free(node->block_table);
    delete node;
}
