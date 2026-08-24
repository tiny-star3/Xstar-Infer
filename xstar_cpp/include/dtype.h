#pragma once
#include <cstddef>
#include <stdexcept>

/**
 * Strongly-typed enum: does not pollute the enclosing namespace and won't implicitly convert to int.
 */
enum class DType
{
    Float32,  // 4 bytes
    BFloat16, // 2 bytes
};

/**
 * Returns the byte width of a dtype. Used to compute nbytes: numel * dtype_size(dtype).
 */
inline std::size_t dtype_size(DType d)
{
    switch (d)
    {
    case DType::Float32:
        return 4;
    case DType::BFloat16:
        return 2;
    }
    // inline 让链接器接受多份定义、合并成一份(头文件内联函数的标准用法)
    // default 抛异常: 未知 dtype 早爆,不静默返回 0
    throw std::runtime_error("unknown DType");
}