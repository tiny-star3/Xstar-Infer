#include <stdexcept>

#include "ops/paged_write.h"
#include "cuda/cuda_allocator.h"

void paged_write(const BlockManager &bm, int layer, const Tensor &K, const Tensor &V, const int *slot_mapping)
{
    if (K.shape().size() != 3 || V.shape().size() != 3)
        throw std::runtime_error("rank mismatch");
    if (K.dtype() != V.dtype())
        throw std::runtime_error("dtype mismatch");
    if (K.shape()[1] != V.shape()[1] || K.shape()[0] != V.shape()[0] || K.shape()[2] != V.shape()[2])
        throw std::runtime_error("shape mismatch");
    if (K.device() != V.device())
        throw std::runtime_error("device mismatch");

    if (K.device() == Device::CUDA)
    {
        std::int64_t seq = K.shape()[K.shape().size() - 2];
        std::int64_t nkv = K.shape()[K.shape().size() - 3];
        std::int64_t hd = K.shape()[K.shape().size() - 1];
        std::size_t slot_mapping_bytes = seq * sizeof(int);
        std::int64_t dz = static_cast<std::int64_t>(dtype_size(K.dtype()));

        void *d_slot_mapping = cuda_alloc(slot_mapping_bytes);
        cuda_memcpy_h2d(d_slot_mapping, slot_mapping, slot_mapping_bytes);
        paged_write_launch(bm.pool_ptr(), K.data(), V.data(), static_cast<const int *>(d_slot_mapping), bm.layer_stride() / dz, bm.block_bytes() / dz, layer, seq, nkv, hd, bm.block_size(), K.dtype());
        cuda_free(d_slot_mapping);
    }
    else
        throw std::runtime_error("unsupported device");
}