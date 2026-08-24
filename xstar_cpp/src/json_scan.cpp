#include <stdexcept>
#include <cstdlib>

#include "json_scan.h"

void skip_ws(const char *&p, const char *end)
{
    // 越界优先: p >= end 时一个字节都不能读, 否则越界 UB
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
        ++p;
}

std::string read_string(const char *&p, const char *end)
{
    if (p == end || *p != '"')
        throw std::runtime_error("p not pointing at '\"'");

    std::string s;
    p++;
    while (p < end && *p != '\"')
    {
        if (*p == '\\')
            throw std::runtime_error("backslash encountered");
        if (*p >= 0x80)
            throw std::runtime_error("non-ASCII byte (>=0x80)");
        s += *p;
        p++;
    }
    if (p == end)
        throw std::runtime_error("closing '\"' not found before end");
    p++;

    return s;
}

std::size_t read_uint(const char *&p, const char *end)
{
    if (p == end)
        throw std::runtime_error("p == end");
    // safetensors 的 shape/data_offsets 按规范非负, 负号 = 文件损坏或扫错位
    if (*p == '-')
        throw std::runtime_error("'-' encountered");
    if (*p < '0' || *p > '9')
        throw std::runtime_error("p not pointing at a digit");

    // 假设数字在 size_t 范围内, 不防 val*10 累加溢出
    std::size_t val = 0;
    while (p < end && *p >= '0' && *p <= '9')
    {
        val = val * 10 + (*p - '0');
        p++;
    }

    return val;
}

std::vector<std::size_t> read_int_array(const char *&p, const char *end)
{
    if (p == end || *p != '[')
        throw std::runtime_error("p not pointing at '['");

    std::vector<std::size_t> result;
    skip_ws(++p, end);
    while (p < end && *p != ']')
    {
        std::size_t val = read_uint(p, end);
        result.push_back(val);
        skip_ws(p, end);
        if (p == end || *p == ']')
        {
            break;
        }
        else if (*p == ',')
        {
            skip_ws(++p, end);
        }
        else
        {
            throw std::runtime_error("unsupported char encountered");
        }
    }
    if (p == end)
        throw std::runtime_error("closing ']' not found before end");
    p++;
    return result;
}

void skip_object(const char *&p, const char *end)
{
    if (p == end || *p != '{')
        throw std::runtime_error("p not pointing at '{'");

    skip_ws(++p, end);
    std::size_t depth = 1;
    bool in_string = false;

    while (p < end && depth != 0)
    {
        // 字符串内的 {} 不计入配对: 维护 in_string, 遇 '"' 翻转, in_string 为真时 {} 忽略
        if (*p == '"')
        {
            in_string ^= 1;
        }
        else if (*p == '{')
        {
            depth += in_string ? 0 : 1;
        }
        else if (*p == '}')
        {
            depth -= in_string ? 0 : 1;
        }
        else if (*p == '\\')
        {
            // 遇 '\' 抛异常: 整个头一律拒绝转义 (与 read_string 对 '\' 态度一致)
            throw std::runtime_error("backslash encountered");
        }
        p++;
    }
    if (depth != 0)
        throw std::runtime_error("closing '}' not found before end");
}

void skip_array(const char *&p, const char *end)
{
    if (p == end || *p != '[')
        throw std::runtime_error("p not pointing at '['");

    skip_ws(++p, end);
    std::size_t depth = 1;
    bool in_string = false;

    while (p < end && depth != 0)
    {
        // 字符串内的 [] 不计入配对: 维护 in_string, 遇 '"' 翻转, in_string 为真时 [] 忽略
        if (*p == '"')
        {
            in_string ^= 1;
        }
        else if (*p == '[')
        {
            depth += in_string ? 0 : 1;
        }
        else if (*p == ']')
        {
            depth -= in_string ? 0 : 1;
        }
        else if (*p == '\\')
        {
            // 遇 '\' 抛异常: 整个头一律拒绝转义 (与 read_string 对 '\' 态度一致)
            throw std::runtime_error("backslash encountered");
        }
        p++;
    }
    if (depth != 0)
        throw std::runtime_error("closing ']' not found before end");
}

double read_number(const char *&p, const char *end)
{
    if (p == end || (*p != '+' && *p != '-' && (*p < '0' || *p > '9')))
        throw std::runtime_error("p not pointing at a digit or '-' or '+");

    const char *end_ptr = p;
    while (end_ptr < end)
    {
        if (*end_ptr != '+' && *end_ptr != '-' && *end_ptr != '.' && *end_ptr != 'e' && *end_ptr != 'E' && (*end_ptr < '0' || *end_ptr > '9'))
        {
            break;
        }
        end_ptr++;
    }

    std::string buf(p, end_ptr);
    char *ptr;
    double result = std::strtod(buf.c_str(), &ptr);
    if (ptr == buf.c_str())
        throw std::runtime_error("the bounded literal contains no digit");
    if (ptr - buf.c_str() != end_ptr - p)
        throw std::runtime_error("strtod does not consume the whole literal (endptr stops short of the buffer end, meaning a malformed number)");
    p = end_ptr;
    return result;
}

bool read_bool(const char *&p, const char *end)
{
    if (p == end || (*p != 't' && *p != 'f'))
        throw std::runtime_error("p not pointing at 't' or 'f");
    if (*p == 't')
    {
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 'r')
            throw std::runtime_error("the bytes after 't' are not exactly \"rue\"");
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 'u')
            throw std::runtime_error("the bytes after 't' are not exactly \"rue\"");
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 'e')
            throw std::runtime_error("the bytes after 't' are not exactly \"rue\"");
        p++;
        return true;
    }
    else
    {
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 'a')
            throw std::runtime_error("the bytes after 'f' are not exactly \"alse\"");
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 'l')
            throw std::runtime_error("the bytes after 'f' are not exactly \"alse\"");
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 's')
            throw std::runtime_error("the bytes after 'f' are not exactly \"alse\"");
        p++;
        if (p == end)
            throw std::runtime_error("the literal is truncated by end");
        if (*p != 'e')
            throw std::runtime_error("the bytes after 'f' are not exactly \"alse\"");
        p++;
        return false;
    }
}