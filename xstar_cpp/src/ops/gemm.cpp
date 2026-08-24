#include <stdexcept>

#include "ops/gemm.h"
#include "bfloat16.h"

Tensor gemm(const Tensor &A, const Tensor &B, bool transB)
{
    if (A.dtype() != B.dtype())
        throw std::runtime_error("dtype mismatch");
    if (A.shape().size() != 2 || B.shape().size() != 2)
        throw std::runtime_error("rank mismatch");
    if (A.shape()[1] != (transB ? B.shape()[1] : B.shape()[0]))
        throw std::runtime_error("inner-dim mismatch");
    if (A.device() != B.device())
        throw std::runtime_error("device mismatch");

    std::int64_t m = A.shape()[0];
    std::int64_t k = A.shape()[1];
    std::int64_t n = transB ? B.shape()[0] : B.shape()[1];
    std::int64_t lda = k;
    std::int64_t ldb = transB ? k : n;
    std::int64_t ldc = n;
    Tensor C(std::vector<std::int64_t>{m, n}, A.dtype(), A.device());

    if (A.device() == Device::CUDA)
    {
        gemm_launch(C.data(), A.data(), B.data(), nullptr, m, k, n, transB, lda, ldb, ldc, A.dtype());

        return C;
    }
    else if (A.device() == Device::CPU)
    {
        if (A.dtype() == DType::Float32)
        {
            gemm_cpu<float>(A.data<float>(), B.data<float>(), C.data<float>(), m, k, n, transB, lda, ldb, ldc);
            return C;
        }
        else if (A.dtype() == DType::BFloat16)
        {
            gemm_cpu<bfloat16>(A.data<bfloat16>(), B.data<bfloat16>(), C.data<bfloat16>(), m, k, n, transB, lda, ldb, ldc);
            return C;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}