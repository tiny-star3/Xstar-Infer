#include <stdexcept>

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
}

RadixTree::~RadixTree()
{
    auto DeleteTree = [](auto &&self, RadixNode *now)
    {
        for (auto node : now->children)
        {
            if (node.second == nullptr)
            {
                return;
            }
            else
            {
                self(self, node.second);
                delete node.second;
            }
        }
    };
    DeleteTree(DeleteTree, root_);
}

std::pair<int, RadixNode *> RadixTree::match_prefix(const std::vector<int> &tokens)
{
    int matched_length = 0;
    RadixNode *now = root_;
    while (now)
    {
        int m_key;
        for (m_key = 0; m_key < now->key.size() && matched_length + m_key < tokens.size(); m_key++)
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
        if (matched_length >= tokens.size())
        {
            return std::make_pair(matched_length / block_size_ * block_size_, now);
        }
        // 余下不足一块, 无法查叉
        if (tokens.size() - matched_length < block_size_)
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
}