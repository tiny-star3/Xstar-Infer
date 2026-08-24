#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * CHECK_CUDA(call) -- uniform CUDA runtime error checking for all .cu files.
 *
 * Usage:        CHECK_CUDA(cudaMalloc(&p, bytes));
 *               CHECK_CUDA(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
 * Any CUDA runtime API returning cudaError_t goes inside the macro.
 *
 * Failure: throws std::runtime_error("CUDA error: <cudaGetErrorString>") -- NOT a bare bad_alloc / abort, so the message names the failing kind (out of memory vs. invalid value) and tests can catch with pytest.raises(RuntimeError, match=...).
 * throw (not abort) is correct for alloc/copy paths where the caller may recover; cuda_free in a destructor deliberately does NOT use this macro (destructor throwing = terminate, and a failed free is unrecoverable anyway).
 *
 * Why do{}while(0): the macro expands to multiple statements; without the wrapper, `if (cond) CHECK_CUDA(x); else ...` would bind `else` to the macro's inner if, not the outer cond -- the classic multi-statement-macro pitfall.
 * do/while(0) makes the whole macro ONE statement.
 *
 * Layering: this header is CUDA-only -- it includes <cuda_runtime.h> and names cudaError_t, so ONLY .cu files (nvcc) include it.
 * It must NOT be pulled into any .h that g++-compiled .cpp files include (e.g. cuda_allocator.h is kept CUDA-free so tensor.cpp stays g++-compilable).
 * Implementation-layer tool, not interface-layer.
 */
#define CHECK_CUDA(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t cuda_err = call;                                                              \
        if (cuda_err != cudaSuccess)                                                              \
        {                                                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err)); \
        }                                                                                         \
    } while (0)
