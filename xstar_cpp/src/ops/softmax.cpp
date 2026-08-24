#include <stdexcept>
#include <cmath>
#include <limits>

#include "ops/softmax.h"
#include "bfloat16.h"

// 线性编号解多维坐标
size_t row_start(size_t row_idx, int64_t dim, const Tensor &x)
{
    size_t result = 0;
    const std::vector<std::int64_t> shape = x.shape();
    const std::vector<std::int64_t> strides = x.strides();
    for (size_t i = shape.size(); i-- > 0;)
    {
        if (i != dim)
        {
            result += (row_idx % shape[i]) * strides[i];
            // 非-dim 轴的组合
            row_idx /= shape[i];
        }
    }
    return result;
}

Tensor softmax(const Tensor &x, int64_t dim)
{
    size_t rank = x.shape().size();
    if (rank < 1)
        throw std::runtime_error("rank mismatch");
    // 当 signed 和 unsigned 同阶(都是 64-bit)比较时, signed 那个被转成 unsigned
    // 无符号类型参与有符号比较时, 比较语义会被隐式重写, 手动转化 int64_t
    if (dim < -static_cast<int64_t>(rank) || dim >= static_cast<int64_t>(rank))
        throw std::runtime_error("dim out of range");
    if (dim < 0)
        dim += rank;
    Tensor result(x.shape(), x.dtype(), x.device());
    if (x.device() == Device::CUDA)
    {
        std::int64_t outer_size = 1;
        std::int64_t dim_size = 1;
        std::int64_t inner_size = 1;
        for (size_t i = 0; i < x.shape().size(); i++)
        {
            if (i < dim)
                outer_size *= x.shape()[i];
            else if (i == dim)
                dim_size = x.shape()[i];
            else
                inner_size *= x.shape()[i];
        }
        softmax_launch(result.data(), x.data(), outer_size, dim_size, inner_size, x.dtype());

        return result;
    }
    else if (x.device() == Device::CPU)
    {
        int64_t stride = x.strides()[dim];
        size_t row_num = x.numel() / x.shape()[dim];
        std::vector<float> max_x(row_num, -std::numeric_limits<float>::infinity());
        std::vector<float> sum_x(row_num, 0.0f);
        if (x.dtype() == DType::Float32)
        {
            const float *x_data = x.data<float>();
            float *result_data = result.data<float>();
            std::vector<size_t> start(row_num);
            for (size_t i = 0; i < row_num; i++)
            {
                start[i] = row_start(i, dim, x);
                for (size_t j = 0; j < x.shape()[dim]; j++)
                {
                    if (max_x[i] < x_data[start[i] + j * stride])
                    {
                        max_x[i] = x_data[start[i] + j * stride];
                    }
                }
            }
            for (size_t i = 0; i < row_num; i++)
            {
                for (size_t j = 0; j < x.shape()[dim]; j++)
                {
                    sum_x[i] += expf(x_data[start[i] + j * stride] - max_x[i]);
                }
            }
            for (size_t i = 0; i < row_num; i++)
            {
                for (size_t j = 0; j < x.shape()[dim]; j++)
                {
                    result_data[start[i] + j * stride] = expf(x_data[start[i] + j * stride] - max_x[i]) / sum_x[i];
                }
            }
            return result;
        }
        else if (x.dtype() == DType::BFloat16)
        {
            const bfloat16 *x_data = x.data<bfloat16>();
            bfloat16 *result_data = result.data<bfloat16>();
            std::vector<size_t> start(row_num);
            for (size_t i = 0; i < row_num; i++)
            {
                start[i] = row_start(i, dim, x);
                for (size_t j = 0; j < x.shape()[dim]; j++)
                {
                    if (max_x[i] < x_data[start[i] + j * stride])
                    {
                        max_x[i] = static_cast<float>(x_data[start[i] + j * stride]);
                    }
                }
            }
            for (size_t i = 0; i < row_num; i++)
            {
                for (size_t j = 0; j < x.shape()[dim]; j++)
                {
                    sum_x[i] += expf(static_cast<float>(x_data[start[i] + j * stride]) - max_x[i]);
                }
            }
            for (size_t i = 0; i < row_num; i++)
            {
                for (size_t j = 0; j < x.shape()[dim]; j++)
                {
                    result_data[start[i] + j * stride] = static_cast<bfloat16>(expf(static_cast<float>(x_data[start[i] + j * stride]) - max_x[i]) / sum_x[i]);
                }
            }
            return result;
        }
        throw std::runtime_error("unsupported dtype");
    }
    else
        throw std::runtime_error("unsupported device");
}