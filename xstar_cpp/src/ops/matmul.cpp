#include <stdexcept>
#include <vector>

#include "ops/matmul.h"
#include "bfloat16.h"
#include "ops/gemm.h"

Tensor matmul(const Tensor &A, const Tensor &B)
{
    return gemm(A, B, false);
}
