#pragma once
#include <cstdint>
#include <string>

/**
 * The subset of Qwen2.5 config.json fields consumed when building and running the model in this project.
 * Fields not listed here (e.g. architectures, torch_dtype, hidden_act) are intentionally ignored by the loader.
 *
 * Type choices follow the JSON source:
 *   - integer fields are stored as std::int64_t;
 *   - rms_norm_eps and rope_theta are stored as double (NOT float): double represents integers up to 2^53 exactly, covering rope_theta=1e6 and all integer fields with no loss;
 *   - tie_word_embeddings is bool: when true the loader reuses embed_tokens for lm_head (the safetensors file then has no lm_head.weight).
 */
struct Qwen2Config
{
    std::int64_t hidden_size;             // 896
    std::int64_t num_attention_heads;     // 14
    std::int64_t num_key_value_heads;     // 2
    std::int64_t num_hidden_layers;       // 24
    std::int64_t intermediate_size;       // 4864
    std::int64_t max_position_embeddings; // 32768
    std::int64_t vocab_size;              // 151936
    double rms_norm_eps;                  // 1e-06
    double rope_theta;                    // 1000000.0
    bool tie_word_embeddings;             // true
};

/**
 * Parse a Qwen2.5 config.json text into a Qwen2Config.
 *
 * Args:
 *   content: the full text of config.json.
 *            The CALLER reads the file into a std::string; this function does no I/O.
 *            config.json is a few hundred bytes, so mmap would be overkill -- a plain string is the right interface (unlike safetensors, whose multi-GB data section justifies mmap).
 *
 * Returns:
 *   A Qwen2Config with ALL 10 fields populated.
 *
 * Throws std::runtime_error on:
 *   - top-level value is not a JSON object (does not start with '{')
 *   - JSON syntax error (unexpected character during scan)
 *   - backslash or non-ASCII byte (>=0x80) in a string (rejected, not swallowed)
 *   - a required field is MISSING (no silent default -- a missing field means the config is for a different model and must fail loudly)
 *   - a required field's value is not a valid number/bool literal (e.g. hidden_size given as a string); detected and thrown by read_number/read_bool, which reject a value whose first byte is not a digit/'+'/'-' (number) or 't'/'f' (bool)
 *   - a number literal cannot be fully consumed by std::strtod
 *
 * Notes:
 *   - Character-level scanner, NOT a general JSON parser.
 *     Only the 10 fields in Qwen2Config are recognized; every other key has its value skipped.
 *     The skip dispatches on the value's first character and handles string, number, bool, array, and object.
 *   - Numbers are read with std::strtod, so "896", "1000000.0", and "1e-06" are all handled uniformly; the result is a double, cast to the field's type(int64 for integer fields, double for the two float fields).
 *   - Trailing commas are accepted (lenient, consistent with the safetensors parser); machine-written configs do not use them, but leniency removes a special-case branch rather than masking corruption.
 *   - Duplicate keys are not detected; the last occurrence wins (consistent with the safetensors parser; configs are machine-written so this does not arise).
 */
Qwen2Config parse_config_json(const std::string &content);
