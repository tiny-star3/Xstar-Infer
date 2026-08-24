#include <stdexcept>
#include <optional>

#include "qwen2_config.h"
#include "json_scan.h"

Qwen2Config parse_config_json(const std::string &content)
{
    const char *base = content.data();
    const char *end = base + content.size();
    const char *p = base;
    skip_ws(p, end);
    if (p == end)
        throw std::runtime_error("empty config");
    if (*p != '{')
        throw std::runtime_error("top-level value is not a JSON object (does not start with '{')");
    // 过顶层 object '{'
    p++;

    std::optional<std::int64_t> hidden_size;
    std::optional<std::int64_t> num_attention_heads;
    std::optional<std::int64_t> num_key_value_heads;
    std::optional<std::int64_t> num_hidden_layers;
    std::optional<std::int64_t> intermediate_size;
    std::optional<std::int64_t> max_position_embeddings;
    std::optional<std::int64_t> vocab_size;
    std::optional<double> rms_norm_eps;
    std::optional<double> rope_theta;
    std::optional<bool> tie_word_embeddings;
    while (true)
    {
        skip_ws(p, end);
        if (p == end)
            throw std::runtime_error("closing '}' not found before end");
        // 顶层 object 结束
        if (*p == '}')
        {
            p++;
            break;
        }
        std::string key = read_string(p, end);
        skip_ws(p, end);
        if (p == end || *p != ':')
            throw std::runtime_error("expected ':' after key");
        p++;
        skip_ws(p, end);
        if (key == "hidden_size")
        {
            hidden_size = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "num_attention_heads")
        {
            num_attention_heads = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "num_key_value_heads")
        {
            num_key_value_heads = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "num_hidden_layers")
        {
            num_hidden_layers = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "intermediate_size")
        {
            intermediate_size = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "max_position_embeddings")
        {
            max_position_embeddings = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "vocab_size")
        {
            vocab_size = static_cast<std::int64_t>(read_number(p, end));
        }
        else if (key == "rms_norm_eps")
        {
            rms_norm_eps = read_number(p, end);
        }
        else if (key == "rope_theta")
        {
            rope_theta = read_number(p, end);
        }
        else if (key == "tie_word_embeddings")
        {
            tie_word_embeddings = read_bool(p, end);
        }
        else
        {
            if (*p == '[')
            {
                skip_array(p, end);
            }
            else if (*p == '{')
            {
                skip_object(p, end);
            }
            else if (*p == '"')
            {
                read_string(p, end);
            }
            else if (*p == '+' || *p == '-' || (*p >= '0' && *p <= '9'))
            {
                read_number(p, end);
            }
            else if (*p == 't' || *p == 'f')
            {
                read_bool(p, end);
            }
            else
            {
                throw std::runtime_error("JSON syntax error (unexpected character during scan)");
            }
        }
        skip_ws(p, end);
        // 字段间逗号 dispatch(同顶层,宽容尾逗号)
        if (p == end)
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
            throw std::runtime_error("expected ',' or '}' between config key");
        }
    }
    if (!hidden_size)
        throw std::runtime_error("missing required config field: hidden_size");
    else if (!num_attention_heads)
        throw std::runtime_error("missing required config field: num_attention_heads");
    else if (!num_key_value_heads)
        throw std::runtime_error("missing required config field: num_key_value_heads");
    else if (!num_hidden_layers)
        throw std::runtime_error("missing required config field: num_hidden_layers");
    else if (!intermediate_size)
        throw std::runtime_error("missing required config field: intermediate_size");
    else if (!max_position_embeddings)
        throw std::runtime_error("missing required config field: max_position_embeddings");
    else if (!vocab_size)
        throw std::runtime_error("missing required config field: vocab_size");
    else if (!rms_norm_eps)
        throw std::runtime_error("missing required config field: rms_norm_eps");
    else if (!rope_theta)
        throw std::runtime_error("missing required config field: rope_theta");
    else if (!tie_word_embeddings)
        throw std::runtime_error("missing required config field: tie_word_embeddings");

    // 顺序须与 Qwen2Config 成员声明顺序一致
    return Qwen2Config{*hidden_size, *num_attention_heads, *num_key_value_heads, *num_hidden_layers, *intermediate_size, *max_position_embeddings, *vocab_size, *rms_norm_eps, *rope_theta, *tie_word_embeddings};
}