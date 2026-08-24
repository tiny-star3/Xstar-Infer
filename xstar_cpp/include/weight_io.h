#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>

#include "mmap_file.h"
#include "tensor.h"

/**
 * Metadata for one tensor parsed from a safetensors header.
 * offset is RELATIVE TO THE FILE START (already includes the 8-byte length prefix + header).
 */
struct TensorMeta
{
    std::size_t offset;
    std::vector<std::int64_t> shape;
    DType dtype;
};

/**
 * Focus solely on "creating the CPU view" and steer clear of device migration, as the output of `mmap` is inherently tied to the CPU.
 */
Tensor make_weight_view(MMapFile &mf, std::size_t offset, std::vector<int64_t> shape, DType dtype);

/**
 * Parse a safetensors file header and return a name -> tensor-metadata lookup.
 *
 * File layout: [8 bytes: u64 LE header_len][header_len bytes: JSON header][tensor data].
 *   JSON is a flat object: {"tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}, ...}.
 *   data_offsets in the JSON are RELATIVE to the start of the tensor-data section;
 *   the returned offset is RELATIVE TO THE FILE START (this function adds 8 + header_len),
 *   so callers pass it directly to make_weight_view without any extra arithmetic.
 *
 * A special "__metadata__" key (value is a nested object, not a tensor) is SKIPPED.
 *
 * Args:
 *   mf: MMapFile of the whole file (read-only). Header and data both live in this mapping.
 *
 * Returns:
 *   std::unordered_map<std::string, TensorMeta>, where TensorMeta = { offset, shape, dtype }.
 *   O(1) lookup by tensor name; one entry per tensor in the file.
 *
 * Throws std::runtime_error on:
 *   - file smaller than 8 bytes (cannot read header_len)
 *   - header_len exceeds remaining file bytes (truncated header)
 *   - JSON syntax error (unexpected character during scan)
 *   - backslash or non-ASCII byte in a string (explicitly rejected, not silently swallowed)
 *   - dtype not "F32" or "BF16" (only these two are supported)
 *
 * Notes:
 *   - This is a CHARACTER-LEVEL SCANNER for the safetensors header shape, NOT a general JSON parser.
 *     No nesting except the __metadata__ object (brace-matched and skipped), no escape sequences, no scientific-notation numbers -- values are only string dtypes and integer arrays (shape, data_offsets).
 *   - Padding spaces in the header are skipped as whitespace.
 *   - Duplicate keys within an object are not detected; the last occurrence wins. safetensors headers are machine-written so this does not arise in practice; detecting it would require a per-object seen-set with no correctness benefit.
 */
std::unordered_map<std::string, TensorMeta> parse_safetensors_header(const MMapFile &mf);
