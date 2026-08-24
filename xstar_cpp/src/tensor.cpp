#include <cstdlib>
#include <stdexcept>
#include <new>

#include "tensor.h"
#include "bfloat16.h"
#include "cuda/cuda_allocator.h"

// 默认值 = Device::CPU 只能在声明(.h)出现一次, 定义(.cpp)不能再写, 否则编译错 "redefinition of default argument"
Tensor::Tensor(std::vector<std::int64_t> shape, DType dtype, Device device) : shape_(std::move(shape)), dtype_(dtype), device_(device), owns_data_(true)
{
    std::int64_t stride = 1;
    strides_ = std::vector<std::int64_t>(shape_.size());
    // size_t 是无符号。循环到 i == 0 后,i-- 变成 SIZE_MAX(不是 -1), i >= 0 永远为真 → 死循环 / 越界写
    // 无符号倒序的惯用法
    for (std::size_t i = strides_.size(); i-- > 0;)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    // 循环结束后 stride == numel
    std::size_t nb = stride * dtype_size(dtype);
    // aligned_alloc(64, nb) 要求 nb 是 alignment(64) 的整数倍, 否则行为未定义
    std::size_t aligned_nb = (nb + 63) & ~63;
    // aligned_alloc 的 size=0 行为未定义, nb==0 时给最小 64
    if (aligned_nb == 0)
        aligned_nb = 64;

    // 成员初始化列表不显式赋 data_,靠 in-class 初始值兜异常安全
    if (device == Device::CPU)
    {
        data_ = std::aligned_alloc(64, aligned_nb);
    }
    else if (device == Device::CUDA)
    {
        data_ = cuda_alloc(aligned_nb);
    }
    else
        throw std::runtime_error("unsupported device");
    if (data_ == nullptr)
        throw std::bad_alloc();
}

Tensor::Tensor(const void *ptr, std::vector<std::int64_t> shape, DType dtype, Device device) : data_(const_cast<void *>(ptr)), shape_(std::move(shape)), dtype_(dtype), device_(device), owns_data_(false)
{
    std::int64_t stride = 1;
    strides_ = std::vector<std::int64_t>(shape_.size());
    // size_t 是无符号。循环到 i == 0 后,i-- 变成 SIZE_MAX(不是 -1), i >= 0 永远为真 → 死循环 / 越界写
    // 无符号倒序的惯用法
    for (std::size_t i = strides_.size(); i-- > 0;)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

Tensor::~Tensor()
{
    release();
}

Tensor::Tensor(Tensor &&other) noexcept : data_(other.data_), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), dtype_(other.dtype_), device_(other.device_), owns_data_(other.owns_data_)
{
    other.data_ = nullptr;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    // 必须判自赋值,否则先 release 自己再从 other 搬会把自己置空
    if (this != &other)
    {
        release();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
    }
    return *this;
}

const std::vector<std::int64_t> &Tensor::shape() const
{
    return shape_;
}

const std::vector<std::int64_t> &Tensor::strides() const
{
    return strides_;
}

DType Tensor::dtype() const
{
    return dtype_;
}

Device Tensor::device() const
{
    return device_;
}

std::int64_t Tensor::numel() const
{
    std::int64_t result = 1;
    for (std::size_t i = 0; i < shape_.size(); i++)
    {
        result *= shape_[i];
    }
    return result;
}

std::size_t Tensor::nbytes() const
{
    return numel() * dtype_size(dtype_);
}

void *Tensor::data()
{
    return data_;
}

const void *Tensor::data() const
{
    return data_;
}

// 目前支持 float 和 bfloat16
// 不支持的类型(如 int) → 链接期报错(未特化), 符合"编译/链接期失败优于运行期"
template <>
float *Tensor::data<float>()
{
    return static_cast<float *>(data_);
}

template <>
bfloat16 *Tensor::data<bfloat16>()
{
    return static_cast<bfloat16 *>(data_);
}

template <>
const float *Tensor::data<float>() const
{
    return static_cast<const float *>(data_);
}

template <>
const bfloat16 *Tensor::data<bfloat16>() const
{
    return static_cast<const bfloat16 *>(data_);
}

void Tensor::release()
{
    if (data_ && owns_data_)
    {
        if (device_ == Device::CPU)
        {
            std::free(data_);
        }
        else if (device_ == Device::CUDA)
        {
            cuda_free(data_);
        }
        else
            // 析构函数里抛异常 = terminate(栈展开期间)
            std::abort();
        data_ = nullptr;
    }
}

Tensor to_cuda(const Tensor &t)
{
    if (t.device() == Device::CUDA)
        throw std::runtime_error("The tensor is already on CUDA");

    Tensor result(t.shape(), t.dtype(), Device::CUDA);
    cuda_memcpy_h2d(result.data(), t.data(), t.nbytes());

    return result;
}

Tensor to_cpu(const Tensor &t)
{
    if (t.device() == Device::CPU)
        throw std::runtime_error("The tensor is already on CPU");

    Tensor result(t.shape(), t.dtype(), Device::CPU);
    cuda_memcpy_d2h(result.data(), t.data(), t.nbytes());

    return result;
}