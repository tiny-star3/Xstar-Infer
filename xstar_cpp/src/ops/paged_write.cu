#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "ops/paged_write.h"
#include "cuda/cuda_check.h"

constexpr int WARPSPERBLOCK = 4;

// trait
// f32 用 float4(16B=4 float)、bf16 用 uint2(8B=4 bf16)
template <typename T>
struct vec_type;
template <>
struct vec_type<float>
{
    using type = float4;
};
template <>
struct vec_type<__nv_bfloat16>
{
    using type = uint2;
};
template <typename T>
using vec_t = typename vec_type<T>::type;

template <typename T, int HEAD_DIM>
__global__ void paged_write_kernel(T *pool,
                                   const T *K, const T *V,
                                   const int *d_slot_mapping,
                                   std::int64_t layer_stride_elems, std::int64_t block_elems, int layer,
                                   int num_tokens, int nkv, int hd, int block_size)
{
    int token_idx = blockIdx.x * WARPSPERBLOCK + threadIdx.x / 32;
    if (token_idx >= num_tokens)
        return;

    int lane = threadIdx.x % 32;
    int lanes_per_head = 32 / nkv; // nkv=2 -> 16, nkv=4 -> 8
    int head = lane / lanes_per_head;

    int slot_id = d_slot_mapping[token_idx];
    int block_id = slot_id / block_size;
    int slot = slot_id % block_size;

    std::int64_t src_offset = head * num_tokens * hd + token_idx * hd;

    // nkv=2,hd=64: lanes_per_head=16, 每lane 4 bf16 -> int2
    // nkv=4,hd=64: lanes_per_head=8,  每lane 8 bf16 -> 2int2
    // nkv=8,hd=64: lanes_per_head=4,  每lane 16 bf16 -> 4int2
    // nkv=2 全 coalesce，nkv>2 每 head 内 coalesce、head 间跨段次优
    int step = sizeof(vec_t<T>) / sizeof(T);
    for (int i = (lane % lanes_per_head) * step; i < HEAD_DIM; i += lanes_per_head * step)
    {
        // reinterpret
        // 被加载的精确地址必须 16 对齐,否则非法访存, hd=64 保证对齐
        vec_t<T> val_k = *reinterpret_cast<const vec_t<T> *>(&K[src_offset + i]);
        vec_t<T> val_v = *reinterpret_cast<const vec_t<T> *>(&V[src_offset + i]);
        *reinterpret_cast<vec_t<T> *>(pool + layer * layer_stride_elems + block_id * block_elems + static_cast<std::int64_t>(head) * block_size * HEAD_DIM + slot * HEAD_DIM + i) = val_k;
        *reinterpret_cast<vec_t<T> *>(pool + layer * layer_stride_elems + block_id * block_elems + static_cast<std::int64_t>(nkv) * block_size * HEAD_DIM + static_cast<std::int64_t>(head) * block_size * HEAD_DIM + slot * HEAD_DIM + i) = val_v;
    }
}

void paged_write_launch(void *pool,
                        const void *K, const void *V,
                        const int *d_slot_mapping,
                        std::int64_t layer_stride_elems, int block_elems, int layer,
                        int num_tokens, int nkv, int hd, int block_size,
                        DType dtype)
{
    // 一个 warp 写一个 token
    dim3 threadPerBlock(WARPSPERBLOCK * 32);
    dim3 blockPerGrid((num_tokens + WARPSPERBLOCK - 1) / WARPSPERBLOCK);

    if (32 % nkv != 0)
        throw std::runtime_error("32 % nkv != 0");

    if (dtype == DType::Float32)
    {
        if (hd == 64)
        {
            paged_write_kernel<float, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(pool), static_cast<const float *>(K), static_cast<const float *>(V), d_slot_mapping, layer_stride_elems, block_elems, layer, num_tokens, nkv, hd, block_size);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (hd == 128)
        {
            throw std::runtime_error("unsupported head_dim");
            // paged_write_kernel<float, 128><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(pool), static_cast<const float *>(K), static_cast<const float *>(V), d_slot_mapping, layer_stride_elems, block_elems, layer, num_tokens, nkv, hd, block_size);
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (hd == 256)
        {
            throw std::runtime_error("unsupported head_dim");
            // paged_write_kernel<float, 256><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(pool), static_cast<const float *>(K), static_cast<const float *>(V), d_slot_mapping, layer_stride_elems, block_elems, layer, num_tokens, nkv, hd, block_size);
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else
            throw std::runtime_error("unsupported head_dim");
    }
    else if (dtype == DType::BFloat16)
    {
        if (hd == 64)
        {
            paged_write_kernel<__nv_bfloat16, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(pool), static_cast<const __nv_bfloat16 *>(K), static_cast<const __nv_bfloat16 *>(V), d_slot_mapping, layer_stride_elems, block_elems, layer, num_tokens, nkv, hd, block_size);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (hd == 128)
        {
            throw std::runtime_error("unsupported head_dim");
            // paged_write_kernel<__nv_bfloat16, 128><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(pool), static_cast<const __nv_bfloat16 *>(K), static_cast<const __nv_bfloat16 *>(V), d_slot_mapping, layer_stride_elems, block_elems, layer, num_tokens, nkv, hd, block_size);
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (hd == 256)
        {
            throw std::runtime_error("unsupported head_dim");
            // paged_write_kernel<__nv_bfloat16, 256><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(pool), static_cast<const __nv_bfloat16 *>(K), static_cast<const __nv_bfloat16 *>(V), d_slot_mapping, layer_stride_elems, block_elems, layer, num_tokens, nkv, hd, block_size);
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else
            throw std::runtime_error("unsupported head_dim");
    }
    else
        throw std::runtime_error("unsupported dtype");
}