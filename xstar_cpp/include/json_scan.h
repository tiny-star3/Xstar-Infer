#pragma once
#include <cstring>
#include <vector>
#include <string>

/**
 * Skip whitespace characters (space, \\n, \\r, \\t) starting at p, advancing p past them.
 * Stops at the first non-whitespace byte (or end). No-op if p already points at non-whitespace.
 */
void skip_ws(const char *&p, const char *end);

/**
 * Read a JSON string (opening '"' through closing '"') into a std::string.
 * p must point at '"'. On return p points just past the closing '"'.
 *
 * Throws std::runtime_error on: p not pointing at '"'; backslash encountered; non-ASCII byte (>=0x80); closing '"' not found before end.
 */
std::string read_string(const char *&p, const char *end);

/**
 * Read a non-negative decimal integer. p must point at a digit '0'-'9'.
 * Accumulates val = val*10 + (c - '0') as std::size_t, stops at the first non-digit.
 * On return p points at the first non-digit byte.
 *
 * Throws std::runtime_error on: p not pointing at a digit; '-' encountered.
 */
std::size_t read_uint(const char *&p, const char *end);

/**
 * Read a JSON array of non-negative integers, e.g. [896, 896] or [0, 3205632].
 * Lenient: a single trailing comma before ] (e.g. [1, 2,]) is accepted; the strict JSON grammar would reject it.
 * p must point at '['. On return p points just past the closing ']'.
 * An empty array [] is allowed (kept general; M5 does not use it).
 *
 * Throws std::runtime_error on: p not pointing at '['; element not a digit; closing ']' not found before end.
 * Note: missing ] and non-digit elements still throw; a trailing comma does not.
 */
std::vector<std::size_t> read_int_array(const char *&p, const char *end);

/**
 * Skip a JSON object by brace-matching, WITHOUT parsing its contents.
 * p must point at '{'. On return p points just past the matching '}'.
 *
 * Throws std::runtime_error on: p not pointing at '{'; backslash encountered; closing '}' not found before end (depth does not return to zero).
 */
void skip_object(const char *&p, const char *end);

/**
 * Skip a JSON array by bracket-matching, WITHOUT parsing its contents.
 * p must point at '['. On return p points just past the matching ']'.
 *
 * Throws std::runtime_error on: p not pointing at '['; backslash encountered; closing ']' not found before end (depth does not return to zero).
 */
void skip_array(const char *&p, const char *end);

/**
 * Read a JSON number literal (integer, decimal, or scientific notation, e.g. 896, 1000000.0, 1e-06) as a double.
 * p must point at the first byte of the literal (a digit '0'-'9' or '-' or '+').
 * On return p points just past the literal.
 *
 * Two stages:
 *  (1) bound the literal by scanning number-legal characters(0-9, '.', 'e'/'E', '+', '-') so it stops at the next ']', ',', or '}';
 *  (2) feed the bounded, null-terminated substring to std::strtod, which does the IEEE-754 conversion.
 *      strtod cannot bound the literal itself (it parses as far as it can into the buffer), so the bounding scan is mandatory.
 *
 * Throws std::runtime_error on: p not pointing at a digit or '-' or '+'; the bounded literal contains no digit; strtod does not consume the whole literal (endptr stops short of the buffer end, meaning a malformed number).
 */
double read_number(const char *&p, const char *end);

/**
 * Read a JSON boolean literal (true/false) as a bool. p must point at 't' or 'f'.
 * On return p points just past the literal (4 bytes for true, 5 for false).
 *
 * Throws std::runtime_error on: p not pointing at 't' or 'f'; the bytes after 't' are not exactly "rue" or the bytes after 'f' are not exactly "alse"; the literal is truncated by end.
 */
bool read_bool(const char *&p, const char *end);