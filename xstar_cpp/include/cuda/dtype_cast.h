#pragma once
#include <type_traits>
#include <cuda_runtime.h>

template <typename T>
__device__ float toFloat(T val)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return val;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return __bfloat162float(val);
    }
    return val;
}