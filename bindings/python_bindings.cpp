#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> //  支持 std::vector(及 list/tuple 等 STL 容器)和 Python list/tuple 的自动互转

#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <memory>
#include <optional>
#include <numeric>

#include "tensor.h"
#include "bfloat16.h"
#include "mmap_file.h"
#include "weight_io.h"

#include "ops/rmsnorm.h"
#include "ops/embedding.h"
#include "ops/rope.h"
#include "ops/matmul.h"
#include "ops/linear.h"
#include "ops/softmax.h"
#include "ops/attention.h"
#include "ops/gemm.h"
#include "ops/mlp.h"
#include "ops/transformer_block.h"

#include "qwen2_config.h"
#include "qwen2_model.h"

#include "cuda/cuda_allocator.h"

#include "ops/concat.h"
#include "ops/head_split.h"
#include "ops/add.h"

#include "block_manager.h"
#include "ops/attention_fa2.h"
#include "kv_cache.h"
#include "paged_kv_cache.h"

#include "radix_cache.h"

namespace py = pybind11;

// 从 numpy 数组构造 Tensor(拷贝数据进 owned buffer)
// 接 py::array(无类型),内部按 dtype 派发
Tensor from_numpy(py::array array)
{
    if (!array.dtype().equal(py::dtype::of<float>()))
        throw std::runtime_error("unsupported dtype");

    std::vector<std::int64_t> shape(array.ndim());
    for (size_t i = 0; i < shape.size(); i++)
    {
        shape[i] = array.shape(i);
    }
    Tensor tensor(shape, DType::Float32, Device::CPU);
    // contiguity（连续性）检查
    if (!(array.flags() & py::array::c_style))
        throw std::runtime_error("array must be C-contiguous");
    std::memcpy(tensor.data(), array.data(), tensor.nbytes());
    return tensor;
}

// 从 Tensor 拷出数据造 numpy 数组返回
py::array to_numpy(const Tensor &t)
{
    if (t.dtype() != DType::Float32)
        throw std::runtime_error("unsupported dtype");

    py::array_t<float> array(t.shape());
    // mutable_data() 明确要可写指针, 避免 data() 的 const 歧义
    std::memcpy(array.mutable_data(), t.data(), t.nbytes());
    // py::array_t<float> 可隐式转 py::array 返回
    return array;
}

// Construct a Tensor by COPYING raw bytes from `array`, interpreted as `dtype`.
// array's numpy dtype carries NO semantic meaning — only its element byte-size must match dtype_size(dtype):
//   Float32  -> numpy float32 (4 bytes/elt)
//   BFloat16 -> numpy uint16  (2 bytes/elt, raw bf16 bits reinterpreted)
// Shape comes from `shape` (like safetensors: layout from metadata, not bytes).
// Throws std::runtime_error if:
//   - array.itemsize() != dtype_size(dtype), or
//   - array.size()    != product(shape)   (byte count mismatch).
Tensor from_numpy_raw(py::array array, std::vector<std::int64_t> shape, DType dtype)
{
    if (array.itemsize() != dtype_size(dtype))
        throw std::runtime_error("itemsize mismatch: array itemsize != dtype size");
    std::int64_t numel = 1;
    for (auto s : shape)
        numel *= s;
    if (array.size() != numel)
        throw std::runtime_error("byte count mismatch");

    Tensor t(shape, dtype, Device::CPU);
    // contiguity（连续性）
    if (!(array.flags() & py::array::c_style))
        throw std::runtime_error("array must be C-contiguous");
    std::memcpy(t.data(), array.data(), t.nbytes());
    return t;
}

// Copy a Tensor's raw bytes out as a flat numpy array of uint8.
// The numpy dtype carries NO semantic meaning — it is raw bytes only:
//   Float32  Tensor -> 4×numel bytes
//   BFloat16 Tensor -> 2×numel bytes (bf16 bits, for the caller to reinterpret)
// Shape is FLAT (1-D, length = nbytes) so callers can reshape/view freely.
// This is the inverse of from_numpy_raw: bits round-trip exactly.
py::array to_numpy_raw(const Tensor &t)
{
    // 1-D uint8 数组, 长度 = t.nbytes()
    py::array_t<std::uint8_t> array(static_cast<py::ssize_t>(t.nbytes()));
    std::memcpy(array.mutable_data(), t.data(), t.nbytes());

    // 隐式转 py::array
    return array;
}

// py::array_t<T, ExtraFlags> 的 ExtraFlags 默认值是 py::array::c_style | py::array::forcecast——默认就带 forcecast, 也就是默认会静默转换非 int64 数组
// py::array::c_style 只传了连续性, 把默认的 forcecast 覆盖掉了——这恰好变成"严格"(不转换)
// c_style flag 已经保证 ids 进来时是连续的(非连续会被拷)
// 拒绝有损转换(float64→int64), 允许安全宽化(int32→int64); 并强制 C-连续(非连续输入会被拷贝成连续)
Tensor embedding_py(Tensor &weight, py::array_t<std::int64_t, py::array::c_style> ids)
{
    const std::int64_t *ids_data = ids.data();
    std::vector<std::int64_t> ids_shape(ids.ndim());
    for (size_t i = 0; i < ids_shape.size(); i++)
    {
        ids_shape[i] = ids.shape(i);
    }
    return embedding(weight, ids_data, ids_shape);
}

// py::array_t<T, ExtraFlags> 的 ExtraFlags 默认值是 py::array::c_style | py::array::forcecast——默认就带 forcecast, 也就是默认会静默转换非 int64 数组
// py::array::c_style 只传了连续性, 把默认的 forcecast 覆盖掉了——这恰好变成"严格"(不转换)
// c_style flag 已经保证 positions 进来时是连续的(非连续会被拷)
// 拒绝有损转换(float64→int64), 允许安全宽化(int32→int64); 并强制 C-连续(非连续输入会被拷贝成连续)
Tensor rope_py(Tensor &x, Tensor &cache, py::array_t<std::int64_t, py::array::c_style> positions)
{
    if (positions.shape(0) != x.shape()[x.shape().size() - 2])
        throw std::runtime_error("positions length != seq_len");
    return rope(x, cache, positions.data());
}

Tensor linear_py(Tensor &x, Tensor &weight, py::object bias_python)
{
    if (bias_python.is_none())
    {
        // 判断是不是 None
        return linear(x, weight, nullptr);
    }
    else
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 bias_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &bias = bias_python.cast<Tensor &>();
        return linear(x, weight, &bias);
    }
}

Tensor attention_py(Tensor &Q, Tensor &K, Tensor &V, py::object mask_python)
{
    if (mask_python.is_none())
    {
        // 判断是不是 None
        return attention(Q, K, V, nullptr);
    }
    else
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 mask_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &mask = mask_python.cast<Tensor &>();
        return attention(Q, K, V, &mask);
    }
}

// py::array_t<T, ExtraFlags> 的 ExtraFlags 默认值是 py::array::c_style | py::array::forcecast——默认就带 forcecast, 也就是默认会静默转换非 int64 数组
// py::array::c_style 只传了连续性, 把默认的 forcecast 覆盖掉了——这恰好变成"严格"(不转换)
// c_style flag 已经保证 positions 进来时是连续的(非连续会被拷)
// 拒绝有损转换(float64→int64), 允许安全宽化(int32→int64); 并强制 C-连续(非连续输入会被拷贝成连续)
Tensor transformer_block_py(Tensor &x,
                            std::int64_t num_heads,
                            Tensor &ln1_w, Tensor &ln2_w, float eps,
                            Tensor &q_w, py::object q_b_python,
                            Tensor &k_w, py::object k_b_python,
                            Tensor &v_w, py::object v_b_python,
                            Tensor &o_w,
                            Tensor &gate_up_w, Tensor &down_w,
                            Tensor &cache, py::array_t<std::int64_t, py::array::c_style> positions,
                            py::object mask_python)
{
    if (positions.shape(0) != x.shape()[x.shape().size() - 2])
        throw std::runtime_error("positions length != seq_len");
    Tensor *q_b_ptr = nullptr;
    Tensor *k_b_ptr = nullptr;
    Tensor *v_b_ptr = nullptr;
    Tensor *mask_ptr = nullptr;
    if (!q_b_python.is_none())
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 q_b_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &q_b = q_b_python.cast<Tensor &>();
        q_b_ptr = &q_b;
    }
    if (!k_b_python.is_none())
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 k_b_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &k_b = k_b_python.cast<Tensor &>();
        k_b_ptr = &k_b;
    }
    if (!v_b_python.is_none())
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 v_b_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &v_b = v_b_python.cast<Tensor &>();
        v_b_ptr = &v_b;
    }
    if (!mask_python.is_none())
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 mask_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &mask = mask_python.cast<Tensor &>();
        mask_ptr = &mask;
    }
    return transformer_block(x, num_heads, ln1_w, ln2_w, eps, q_w, q_b_ptr, k_w, k_b_ptr, v_w, v_b_ptr, o_w, gate_up_w, down_w, cache, positions.data(), mask_ptr);
}

// py::array_t<T, ExtraFlags> 的 ExtraFlags 默认值是 py::array::c_style | py::array::forcecast——默认就带 forcecast, 也就是默认会静默转换非 int64 数组
// py::array::c_style 只传了连续性, 把默认的 forcecast 覆盖掉了——这恰好变成"严格"(不转换)
// c_style flag 已经保证 input_ids 进来时是连续的(非连续会被拷)
// 拒绝有损转换(float64→int64), 允许安全宽化(int32→int64); 并强制 C-连续(非连续输入会被拷贝成连续)
Tensor qwen2_forward_py(Qwen2ModelWeights &w, Qwen2Config &cfg, Tensor &rope_cache, py::array_t<std::int64_t, py::array::c_style> input_ids, py::object positions_python, py::object mask_python)
{
    std::int64_t seq_len = input_ids.shape(0);
    const std::int64_t *positions_ptr = nullptr;
    Tensor *mask_ptr = nullptr;
    py::array_t<std::int64_t, py::array::c_style> positions;
    // 空构造,不分配
    std::optional<std::vector<std::int64_t>> buf;
    if (!positions_python.is_none())
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 positions_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        positions = positions_python.cast<py::array_t<std::int64_t, py::array::c_style>>();
        if (positions.shape(0) != seq_len)
            throw std::runtime_error("positions length != seq_len");
        // .data() 指针依赖 positions 持有 numpy buffer, 在调用前保持活着
        positions_ptr = positions.data();
    }
    else
    {
        // positions 为 None, 造一个 arange(seq_len) 的 int64 buffer
        buf = std::vector<int64_t>(seq_len);
        std::iota(buf->begin(), buf->end(), 0);
        positions_ptr = buf->data();
    }
    if (!mask_python.is_none())
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 mask_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &mask = mask_python.cast<Tensor &>();
        mask_ptr = &mask;
    }
    return qwen2_forward(w, cfg, rope_cache, input_ids.data(), seq_len, positions_ptr, mask_ptr);
}

Tensor concat_py(py::list inputs_python, int axis)
{
    std::vector<const Tensor *> inputs;
    inputs.reserve(inputs_python.size());
    for (auto input : inputs_python)
    {
        const Tensor &t = input.cast<const Tensor &>();
        inputs.push_back(&t);
    }
    return concat(inputs, axis);
}

Tensor attention_fa2_py(Tensor &Q, Tensor &K, Tensor &V, py::object mask_python)
{
    if (mask_python.is_none())
    {
        // 判断是不是 None
        return attention_fa2(Q, K, V, nullptr);
    }
    else
    {
        // 不拷贝, 直接返回指向 Python wrapper 内部那个 C++ Tensor 的引用
        // 只要 mask_python 还活着, Python 那边的 Tensor 对象就活着, 引用就有效
        Tensor &mask = mask_python.cast<Tensor &>();
        return attention_fa2(Q, K, V, &mask);
    }
}

Tensor qwen2_forward_incremental_py(Qwen2ModelWeights &w, Qwen2Config &cfg, Tensor &rope_cache, KVCache &kv_cache, bool is_decode, py::array_t<std::int64_t, py::array::c_style> input_ids, py::object mask_python)
{
    std::int64_t seq_len = input_ids.shape(0);
    const std::int64_t *positions_ptr = nullptr;
    Tensor *mask_ptr = nullptr;
    std::optional<std::vector<std::int64_t>> buf;

    // 路 X:从 kv_cache.cursor() 造 positions(decode: [cursor]; prefill: arange(0, seq_len))
    buf = std::vector<int64_t>(seq_len);
    if (is_decode)
    {
        // decode: 新 token 的绝对位置 = cursor(此步前已存的 token 数) seq_len==1, positions = [cursor]
        (*buf)[0] = kv_cache.cursor();
    }
    else
    {
        // prefill: arange(0, seq_len)
        std::iota(buf->begin(), buf->end(), 0);
    }
    positions_ptr = buf->data();

    // mask: incremental 路径固定 nullptr(prefill causal 现场建; decode 免 mask) 忽略 mask_python,或保留 None 检查后强制 nullptr
    if (!mask_python.is_none())
        throw std::runtime_error("incremental forward does not accept mask (prefill causal on the fly; decode mask-free)");
    // mask_ptr 保持 nullptr

    return qwen2_forward(w, cfg, rope_cache, kv_cache, is_decode,
                         input_ids.data(), seq_len, positions_ptr, mask_ptr);
}

Tensor qwen2_forward_paged_py(Qwen2ModelWeights &w, Qwen2Config &cfg, Tensor &rope_cache, BlockManager &bm, PagedKVCache &kv_cache, bool is_decode, py::array_t<std::int64_t, py::array::c_style> input_ids, py::object mask_python)
{
    std::int64_t seq_len = input_ids.shape(0);
    const std::int64_t *positions_ptr = nullptr;
    Tensor *mask_ptr = nullptr;
    std::optional<std::vector<std::int64_t>> buf;

    // 从 kv_cache.cursor() 造 positions(decode: [cursor]; prefill: arange(0, seq_len))
    buf = std::vector<int64_t>(seq_len);
    if (is_decode)
    {
        // decode: 新 token 的绝对位置 = cursor(此步前已存的 token 数) seq_len==1, positions = [cursor]
        (*buf)[0] = kv_cache.cursor();
    }
    else
    {
        // 支持直接 prefill 和前缀匹配后的 prefill
        std::iota(buf->begin(), buf->end(), kv_cache.cursor());
    }
    positions_ptr = buf->data();

    // mask: incremental 路径固定 nullptr(prefill causal 现场建; decode 免 mask) 忽略 mask_python,或保留 None 检查后强制 nullptr
    if (!mask_python.is_none())
        throw std::runtime_error("paged forward does not accept mask (prefill causal on the fly; decode mask-free)");
    // mask_ptr 保持 nullptr

    return qwen2_forward(w, cfg, rope_cache, bm, kv_cache, is_decode,
                         input_ids.data(), seq_len, positions_ptr, mask_ptr);
}

Tensor qwen2_forward_multi_py(Qwen2ModelWeights &w, Qwen2Config &cfg, Tensor &rope_cache, BlockManager &bm, py::list kv_caches_py, bool is_decode, py::array_t<std::int64_t, py::array::c_style> input_ids, py::array_t<std::int64_t, py::array::c_style> cu_seqlens_q_host_py)
{
    // forward 自己 build positions
    int num_seqs = kv_caches_py.size();
    if (cu_seqlens_q_host_py.shape(0) != num_seqs + 1)
        throw std::runtime_error("cu_seqlens_q_host_py.shape(0) != num_seqs+1");
    if ((std::int64_t)input_ids.shape(0) != cu_seqlens_q_host_py.at(num_seqs))
        throw std::runtime_error("input_ids length != sum_q");
    std::int64_t sum_q = cu_seqlens_q_host_py.at(num_seqs);

    // 转 std::vector<PagedKVCache*>
    std::vector<PagedKVCache *> kv_caches(num_seqs);
    for (int i = 0; i < num_seqs; i++)
    {
        // class_<PagedKVCache> 是默认 unique_ptr holder
        // py::list 的元素是 handle/object
        // .cast<PagedKVCache&>() 会拿到底层 unique_ptr 的解引用, & 就是裸指针
        kv_caches[i] = &kv_caches_py[i].cast<PagedKVCache &>();
    }

    // 转 std::vector<int64_t>
    std::vector<int64_t> cu_seqlens_q_host(num_seqs + 1);
    for (int i = 0; i <= num_seqs; i++)
    {
        cu_seqlens_q_host[i] = cu_seqlens_q_host_py.at(i);
    }

    return qwen2_forward(w, cfg, rope_cache, bm, kv_caches, is_decode, input_ids.data(), cu_seqlens_q_host);
}

PYBIND11_MODULE(xstar_cpp, m)
{
    m.doc() = "Xstar C++ tensor runtime";

    // 枚举
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("BFloat16", DType::BFloat16)
        .export_values();
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    // Tensor 类
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<std::int64_t>, DType, Device>(),
             py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU)
        .def("numel", &Tensor::numel)
        .def("nbytes", &Tensor::nbytes)
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype);

    // MMapFile 类
    py::class_<MMapFile>(m, "MMapFile")
        .def(py::init<const std::string &>(),
             py::arg("path"))
        .def("addr", &MMapFile::addr)
        .def("size", &MMapFile::size);

    // TensorMeta 结构体
    // def_readonly 是 pybind11 暴露数据成员的 API(只读,Python 侧能读不能写)
    py::class_<TensorMeta>(m, "TensorMeta")
        .def_readonly("offset", &TensorMeta::offset)
        .def_readonly("shape", &TensorMeta::shape)
        .def_readonly("dtype", &TensorMeta::dtype);

    // Qwen2Config 结构体
    // def_readonly 是 pybind11 暴露数据成员的 API(只读,Python 侧能读不能写)
    py::class_<Qwen2Config>(m, "Qwen2Config")
        .def_readonly("hidden_size", &Qwen2Config::hidden_size)
        .def_readonly("num_attention_heads", &Qwen2Config::num_attention_heads)
        .def_readonly("num_key_value_heads", &Qwen2Config::num_key_value_heads)
        .def_readonly("num_hidden_layers", &Qwen2Config::num_hidden_layers)
        .def_readonly("intermediate_size", &Qwen2Config::intermediate_size)
        .def_readonly("max_position_embeddings", &Qwen2Config::max_position_embeddings)
        .def_readonly("vocab_size", &Qwen2Config::vocab_size)
        .def_readonly("rms_norm_eps", &Qwen2Config::rms_norm_eps)
        .def_readonly("rope_theta", &Qwen2Config::rope_theta)
        .def_readonly("tie_word_embeddings", &Qwen2Config::tie_word_embeddings);

    // Qwen2LayerWeights 结构体
    // Qwen2LayerWeights: 裸 class_, 不暴露字段访问(def_property/def_readwrite 都用不了)
    // 原因: 含 Tensor, Tensor 禁拷贝(防 double-free)且无默认构造 -> 本 struct 也 move-only + 无默认构造, def_readwrite 需拷贝赋值、def_property setter 需 move(但 Python 侧填字段会掏空原 Tensor,语义危险)
    // 对象由 C++ 侧 load_qwen2_weights 构造、move 进 Python; Python 只持有 + 传给 qwen2_forward, 不构造/不拷贝/不读写字段。所以裸绑定足矣, 无需 property
    py::class_<Qwen2LayerWeights>(m, "Qwen2LayerWeights");

    // Qwen2ModelWeights 结构体
    // Qwen2ModelWeights: 同 Qwen2LayerWeights, move-only + 无默认构造, 裸绑定
    // Python 侧只能持有 load_qwen2_weights 的返回值,不能 Qwen2ModelWeights() 构造、不能赋值
    // layers 是 std::vector<Qwen2LayerWeights> —— pybind11 能自动转 Python list
    py::class_<Qwen2ModelWeights>(m, "Qwen2ModelWeights");

    // BlockManager 类
    py::class_<BlockManager>(m, "BlockManager")
        .def(py::init<int, int, int, Device, int>(),
             py::arg("num_blocks"), py::arg("block_size"), py::arg("kv_slot_bytes"), py::arg("dev") = Device::CUDA, py::arg("num_layers") = 1)
        .def("alloc", &BlockManager::alloc)
        .def("free", &BlockManager::free)
        .def("ref", &BlockManager::ref)
        .def("fork", &BlockManager::fork)
        .def("write_block", &BlockManager::write_block)
        .def("cow_copy", &BlockManager::cow_copy)
        .def("num_free", &BlockManager::num_free)
        .def("num_allocated", &BlockManager::num_allocated)
        .def("block_ref_cnt", &BlockManager::block_ref_cnt);

    // BlockTable 结构体
    py::class_<BlockTable>(m, "BlockTable")
        .def_readwrite("physical_ids", &BlockTable::physical_ids);

    // KVCache 类
    py::class_<KVCache>(m, "KVCache")
        .def(py::init<std::int64_t, std::int64_t, std::int64_t, std::int64_t, DType, Device>(),
             py::arg("num_layers"), py::arg("num_kv_heads"),
             py::arg("max_seq_len"), py::arg("head_dim"),
             py::arg("dtype"), py::arg("device") = Device::CUDA)
        .def("cursor", &KVCache::cursor);

    // PagedKVCache 类
    py::class_<PagedKVCache>(m, "PagedKVCache")
        .def(py::init<std::int64_t, std::int64_t, std::int64_t, std::int64_t, DType, Device>(),
             py::arg("num_kv_heads"), py::arg("head_dim"),
             py::arg("max_seq_len"), py::arg("block_size"),
             py::arg("dtype"), py::arg("device") = Device::CUDA)
        .def("cursor", &PagedKVCache::cursor)
        .def("block_table", &PagedKVCache::block_table)
        .def("block_size", &PagedKVCache::block_size)
        .def("reset", &PagedKVCache::reset)
        .def("adopt_prefix", &PagedKVCache::adopt_prefix);

    // RadixNode 结构体 (readonly, 测试用)
    py::class_<RadixNode>(m, "RadixNode")
        .def_readonly("key", &RadixNode::key)
        .def_readonly("block_table", &RadixNode::block_table)
        .def_readonly("lock_ref", &RadixNode::lock_ref)
        .def_readonly("in_lru", &RadixNode::in_lru);

    // RadixTree 类
    py::class_<RadixTree>(m, "RadixTree")
        .def(py::init<int>()) // block_size
        .def("match_prefix", [](RadixTree &self, const std::vector<int> &tokens)
             {
        auto [blocks, node] = self.match_prefix(tokens);
        return py::make_tuple(blocks, py::cast(node, py::return_value_policy::reference)); }, py::arg("tokens")) // reference 政策返回内部指针、不转移所有权
        .def("insert", [](RadixTree &self, const std::vector<int> &tokens, const std::vector<int> &block_table, BlockManager &bm)
             {
        RadixNode *node = self.insert(tokens, block_table, bm);
        return py::cast(node, py::return_value_policy::reference); }, py::arg("tokens"), py::arg("block_table"), py::arg("bm")) // reference 政策返回内部指针、不转移所有权

        .def("inc_lock_ref", &RadixTree::inc_lock_ref)
        .def("dec_lock_ref", &RadixTree::dec_lock_ref)
        .def("evict", &RadixTree::evict)
        .def("lru_size", &RadixTree::lru_size)
        .def("evictable_blocks", &RadixTree::evictable_blocks);

    // 自由函数
    // pybind11/stl.h 自动转 std::vector<Tensor> ↔ Python list[Tensor]
    m.def("from_numpy", &from_numpy, py::arg("array"));
    m.def("to_numpy", &to_numpy, py::arg("tensor"));
    m.def("make_weight_view", &make_weight_view, py::arg("mf"), py::arg("offset"), py::arg("shape"), py::arg("dtype"));
    m.def("from_numpy_raw", &from_numpy_raw, py::arg("array"), py::arg("shape"), py::arg("dtype"));
    m.def("to_numpy_raw", &to_numpy_raw, py::arg("tensor"));
    m.def("parse_safetensors_header", &parse_safetensors_header, py::arg("mf"));

    m.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("weight"), py::arg("eps"));
    m.def("embedding", &embedding_py, py::arg("weight"), py::arg("ids"));
    m.def("rope", &rope_py, py::arg("x"), py::arg("cache"), py::arg("positions"));
    m.def("matmul", &matmul, py::arg("A"), py::arg("B"));
    m.def("linear", &linear_py, py::arg("x"), py::arg("weight"), py::arg("bias"));
    m.def("softmax", &softmax, py::arg("x"), py::arg("dim"));
    m.def("attention", &attention_py, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("mask"));
    m.def("gemm", &gemm, py::arg("A"), py::arg("B"), py::arg("transB"));
    m.def("mlp", &mlp, py::arg("x"), py::arg("gate_up_weight"), py::arg("down_weight"));
    m.def("transformer_block", &transformer_block_py, py::arg("x"), py::arg("num_heads"), py::arg("ln1_w"), py::arg("ln2_w"), py::arg("eps"), py::arg("q_w"), py::arg("q_b"), py::arg("k_w"), py::arg("k_b"), py::arg("v_w"), py::arg("v_b"), py::arg("o_w"), py::arg("gate_up_w"), py::arg("down_w"), py::arg("cache"), py::arg("positions"), py::arg("mask"));

    m.def("parse_config_json", &parse_config_json, py::arg("content"));
    m.def("qwen2_forward", &qwen2_forward_py, py::arg("w"), py::arg("cfg"), py::arg("rope_cache"), py::arg("input_ids"), py::arg("positions"), py::arg("mask"));
    m.def("load_qwen2_weights", &load_qwen2_weights, py::arg("mf"), py::arg("cfg"), py::arg("dev"));

    m.def("to_cuda", &to_cuda, py::arg("t"));
    m.def("to_cpu", &to_cpu, py::arg("t"));
    m.def("cuda_free_bytes", &cuda_free_bytes);
    m.def("concat", &concat_py, py::arg("inputs"), py::arg("axis"));
    m.def("head_split", &head_split, py::arg("t"), py::arg("heads"));
    m.def("add", &add, py::arg("a"), py::arg("b"));

    m.def("attention_fa2", &attention_fa2_py, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("mask"));
    m.def("qwen2_forward_incremental", &qwen2_forward_incremental_py, py::arg("w"), py::arg("cfg"), py::arg("rope_cache"), py::arg("kv_cache"), py::arg("is_decode"), py::arg("input_ids"), py::arg("mask") = py::none());
    m.def("qwen2_forward_paged", &qwen2_forward_paged_py, py::arg("w"), py::arg("cfg"), py::arg("rope_cache"), py::arg("bm"), py::arg("kv_cache"), py::arg("is_decode"), py::arg("input_ids"), py::arg("mask") = py::none());
    m.def("qwen2_forward_multi", &qwen2_forward_multi_py, py::arg("w"), py::arg("cfg"), py::arg("rope_cache"), py::arg("bm"), py::arg("kv_caches"), py::arg("is_decode"), py::arg("input_ids"), py::arg("cu_seqlens_q_host"));
}
