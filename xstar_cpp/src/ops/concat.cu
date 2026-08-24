#include <cuda_runtime.h>

#include "ops/concat.h"
#include "cuda/cuda_check.h"

constexpr int THREADPERBLOCK = 256;

__global__ void concat_kernel(void *out,
                              void **d_ptrs,
                              const std::int64_t *d_axis_sizes,
                              const std::int64_t *d_out_shape,
                              int n,
                              int rank,
                              int axis,
                              int dtype_size,
                              std::int64_t out_numel)
{
    std::int64_t idx = static_cast<std::int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx < out_numel)
    {
        std::int64_t dim = out_numel;
        out = static_cast<char *>(out) + idx * dtype_size;
        std::int64_t coord[MAX_RANK];
        for (int i = 0; i < rank; i++)
        {
            dim /= d_out_shape[i];
            coord[i] = idx / dim;
            idx %= dim;
        }
        dim = coord[axis];
        int in_idx = 0;
        for (int i = 0; i < n; i++)
        {
            if (dim < d_axis_sizes[i])
            {
                coord[axis] = dim;
                in_idx = i;
                break;
            }
            dim -= d_axis_sizes[i];
        }
        char *input_ptr = static_cast<char *>(d_ptrs[in_idx]);
        std::int64_t strides = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            input_ptr += coord[i] * strides * dtype_size;
            strides *= i == axis ? d_axis_sizes[in_idx] : d_out_shape[i];
        }
        memcpy(static_cast<void *>(out), static_cast<void *>(input_ptr), dtype_size);
    }
}

void concat_launch(void *out,
                   void **d_ptrs,
                   const std::int64_t *d_axis_sizes,
                   const std::int64_t *d_out_shape,
                   int n,
                   int rank,
                   int axis,
                   int dtype_size,
                   std::int64_t out_numel)
{
    dim3 threadsPerBlock(THREADPERBLOCK);
    dim3 blockPerGrid((out_numel + threadsPerBlock.x - 1) / threadsPerBlock.x);
    concat_kernel<<<blockPerGrid, threadsPerBlock>>>(out, d_ptrs, d_axis_sizes, d_out_shape, n, rank, axis, dtype_size, out_numel);
    // launch-time 错(不等)
    CHECK_CUDA(cudaGetLastError());
    // 执行错 + 等 kernel 完
    CHECK_CUDA(cudaDeviceSynchronize());
}