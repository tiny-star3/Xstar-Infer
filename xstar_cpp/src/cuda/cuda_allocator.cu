#include <cuda_runtime.h>
#include <new>

#include "cuda/cuda_allocator.h"
#include "cuda/cuda_check.h"

void *cuda_alloc(std::size_t bytes)
{
    void *ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, bytes));

    return ptr;
}

void cuda_free(void *ptr)
{
    if (ptr != nullptr)
    {
        // 内存泄漏或 double-free, 调用方已经没法补救, 析构里抛异常违法, 忽略 cuda_free 失败
        cudaFree(ptr);
    }
}

void cuda_memcpy_h2d(void *dst, const void *src, std::size_t bytes)
{
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void *dst, const void *src, std::size_t bytes)
{
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_d2d(void *dst, const void *src, std::size_t bytes)
{
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

std::size_t cuda_free_bytes()
{
    std::size_t free, total;
    CHECK_CUDA(cudaMemGetInfo(&free, &total));

    return free;
}