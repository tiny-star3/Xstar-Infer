#include <stdexcept>
#include <optional>
#include <cstring>

#include "weight_io.h"
#include "json_scan.h"

Tensor make_weight_view(MMapFile &mf, std::size_t offset, std::vector<int64_t> shape, DType dtype)
{
    // offset 对齐——bf16 要 2 字节、float32 要 4 字节对齐
    if (offset % dtype_size(dtype) != 0)
        throw std::runtime_error("offset not aligned to dtype size");

    // 校验"视图落在映射区内", 防止越界
    std::size_t numel = 1;
    for (auto num : shape)
    {
        numel *= static_cast<std::size_t>(num);
    }
    std::size_t nbytes = numel * dtype_size(dtype);
    // offset 是 std::size_t、nbytes 也是 std::size_t, 两者相加理论上可能溢出(绕回小值), 那就漏判越界了
    // 用减法避免加法溢出
    if ((offset > mf.size()) || (nbytes > mf.size() - offset))
        throw std::runtime_error("weight view exceeds mmap region");

    // 转 const char*, 让 + offset 的单位变成字节
    const char *data = static_cast<const char *>(mf.addr()) + offset;
    // mmap 映射进进程虚拟内存, 就是 CPU 地址
    Tensor tensor = Tensor(static_cast<const void *>(data), std::move(shape), dtype, Device::CPU);
    return tensor;
}

std::unordered_map<std::string, TensorMeta> parse_safetensors_header(const MMapFile &mf)
{
    if (mf.size() < 8)
        throw std::runtime_error("file smaller than 8 bytes (cannot read header_len)");

    const char *base = static_cast<const char *>(mf.addr());
    const char *end = base + mf.size();

    std::uint64_t header_len;
    // 前 8 字节是 u64 小端
    std::memcpy(&header_len, base, 8);

    // 加法可能溢出
    if (header_len > mf.size() - 8)
        throw std::runtime_error("header_len exceeds remaining file bytes (truncated header)");
    const char *header_begin = base + 8;
    const char *header_end = header_begin + header_len;

    // 数据段起点,offset 换算基准
    const char *data_start = header_end;

    const char *p = header_begin;
    skip_ws(p, header_end);
    if (p == header_end)
        throw std::runtime_error("empty header");
    if (*p != '{')
        throw std::runtime_error("top-level is not a JSON object");
    // 过顶层 object '{'
    p++;

    std::unordered_map<std::string, TensorMeta> result;
    while (true)
    {
        skip_ws(p, header_end);
        if (p == header_end)
            throw std::runtime_error("closing '}' not found before end");
        // 顶层 object 结束
        if (*p == '}')
        {
            p++;
            break;
        }
        std::string key = read_string(p, header_end);
        skip_ws(p, header_end);
        if (p == header_end || *p != ':')
            throw std::runtime_error("expected ':' after key");
        p++;
        skip_ws(p, header_end);

        // 分派:value 是 __metadata__ 跳过,否则读 tensor 描述
        if (key == "__metadata__")
        {
            skip_object(p, header_end);
        }
        else
        {
            // tensor descriptor: fixed 3 fields (dtype/shape/data_offsets)
            if (p == header_end)
                throw std::runtime_error("closing '}' not found before end");
            if (*p != '{')
                throw std::runtime_error("tensor descriptor is not a JSON object");
            p++;

            // 读 tensor 描述 { "dtype":..., "shape":..., "data_offsets":[start,end] }
            // 用来"字段可能没出现"
            std::optional<DType> dtype;
            std::optional<std::vector<std::int64_t>> shape;
            // data_offsets 两个元素
            std::optional<std::size_t> data_start_off, data_end_off;
            while (true)
            {
                skip_ws(p, header_end);
                if (p == header_end)
                    throw std::runtime_error("closing '}' not found before end");
                if (*p == '}')
                {
                    p++;
                    break;
                }

                std::string field = read_string(p, header_end);
                skip_ws(p, header_end);
                if (p == header_end || *p != ':')
                    throw std::runtime_error("expected ':' after field");
                p++;
                skip_ws(p, header_end);
                if (field == "dtype")
                {
                    std::string d = read_string(p, header_end);
                    if (d == "F32")
                        dtype = DType::Float32;
                    else if (d == "BF16")
                        dtype = DType::BFloat16;
                    else
                        throw std::runtime_error("dtype not \"F32\" or \"BF16\" (only these two are supported)");
                }
                else if (field == "shape")
                {
                    auto arr = read_int_array(p, header_end);
                    // size_t → int64_t 显式构造转换
                    shape = std::vector<std::int64_t>(arr.begin(), arr.end());
                }
                else if (field == "data_offsets")
                {
                    auto arr = read_int_array(p, header_end);
                    // data_offsets 恒两个元素
                    if (arr.size() != 2)
                        throw std::runtime_error("data_offsets must have exactly 2 elements");
                    data_start_off = arr[0];
                    data_end_off = arr[1];
                }
                else
                {
                    throw std::runtime_error("unsupported field encountered");
                }

                skip_ws(p, header_end);
                // 字段间逗号 dispatch(同顶层,宽容尾逗号)
                if (p == header_end)
                    throw std::runtime_error("closing '}' not found before end");
                if (*p == ',')
                {
                    p++;
                    continue;
                }
                else if (*p == '}')
                {
                    p++;
                    break;
                }
                else
                {
                    throw std::runtime_error("expected ',' or '}' between tensor descriptor fields");
                }
            }
            if (!dtype || !shape || !data_start_off || !data_end_off)
                throw std::runtime_error("tensor descriptor missing required field (dtype/shape/data_offsets)");
            // 一致性校验， (end - start) 应该 == numel × dtype_size
            std::size_t numel = 1;
            for (auto num : *shape)
            {
                numel *= static_cast<std::size_t>(num);
            }
            std::size_t nbytes = numel * dtype_size(*dtype);
            std::size_t data_offsets = *data_end_off - *data_start_off;
            if (nbytes != data_offsets)
                throw std::runtime_error("header inconsistent: data_offsets span " + std::to_string(data_offsets) + " bytes but shape*dtype expects " + std::to_string(nbytes));
            // data_start_off 是相对数据段起点的偏移
            // offset 换算成相对文件起点, 加法焊死在解析器, make_weight_view 不用再算
            std::size_t offset = (data_start - base) + *data_start_off;
            result[key] = {offset, *shape, *dtype};
        }
        skip_ws(p, header_end);
        if (p == header_end)
            throw std::runtime_error("closing '}' not found before end");
        // 宽容: 尾逗号也放行, 逗号后可能直接是 '}',下一轮循环处理
        if (*p == ',')
        {
            p++;
            continue;
        }
        else if (*p == '}')
        {
            p++;
            break;
        }
        else
        {
            throw std::runtime_error("JSON syntax error (unexpected character during scan)");
        }
    }

    return result;
}