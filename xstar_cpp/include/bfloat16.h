#pragma once
#include <cstdint>

/**
 * 16-bit brain float: 1 sign + 8 exponent + 7 mantissa bits.
 * Shares the exponent range of float32; trades mantissa precision for half the storage.
 */
struct bfloat16
{
    std::uint16_t bits; // 16 位存储: 1 符号 + 8 指数 + 7 尾数

    // Construct from float: f32 -> bf16 with round-to-nearest-even truncation.
    bfloat16(float f);

    // Default-construct (uninitialized), for use in containers like std::vector.
    bfloat16() = default;

    // Implicit conversion to float: bf16 -> f32 by zero-extending the low 16 bits (lossless).
    operator float() const;
};
