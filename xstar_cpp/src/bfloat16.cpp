#include <cstring>

#include "bfloat16.h"

bfloat16::bfloat16(float f)
{
    std::uint32_t temp;
    std::memcpy(&temp, &f, sizeof(f));

    // RNE, 看低 16 位的最高位
    // 低 16 位不到一半, 舍掉(不进位)
    // 低 16 位超过一半, 进位(高 16 位 +1)
    // 低 16 位恰好一半: 这是 tie。RNE 规则——向偶数舍入, 看高 16 位最低位(即保留位的最低位), 是 0 就不进, 是 1 就进
    // 目的是让进位后最低位变 0(偶数)

    // 保留位的最低位(决定 tie 向偶)
    std::uint32_t lsb = (temp >> 16) & 1;
    std::uint32_t rounding_bias = 0x7FFF + lsb;
    // 在 uint32 上加(进位自然留在第16位, 不回绕)
    temp += rounding_bias;
    bits = static_cast<uint16_t>(temp >> 16);

    // 已知边界(Phase 1 不处理): NaN/Inf 区段(指数全1)截断时, bf16 尾数可能退化成 0(Inf) 或丢失 NaN 性质; 全 1+1 进位无论 uint16/uint32 都会溢出
    // 真正兜底需显式特判指数全1(保尾数非零或 clamp), 非靠容器宽度。正常推理不触发 NaN
}

bfloat16::operator float() const
{
    std::uint32_t temp;
    temp = bits;
    temp <<= 16;
    float result;
    std::memcpy(&result, &temp, sizeof(result));

    return result;
}