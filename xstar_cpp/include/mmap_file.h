#pragma once
#include <string>
#include <cstdint>

/**
 * RAII owner of a read-only, whole-file memory mapping (mirrors llama.cpp's llama_mmap).
 * Constructor maps the entire file; destructor unmaps. Non-copyable AND non-movable (a user-declared destructor suppresses the implicit move ops; moving would also invalidate the borrowed tensor views that point into the mapping, so it is unwanted even if it compiled).
 * Exposes only the base address + byte length; tensor views are built externally by pointer arithmetic (addr + offset), NOT by a method here — matching llama_mmap.
 */
class MMapFile
{
public:
    // Open `path` O_RDONLY, fstat for size, mmap(PROT_READ, MAP_SHARED) the whole file.
    // Throws std::runtime_error if open/fstat/mmap fails (addr == MAP_FAILED).
    explicit MMapFile(const std::string &path);

    // munmap(addr_, size_). (fd already closed after mmap; the mapping outlives the fd.)
    ~MMapFile();

    MMapFile(const MMapFile &) = delete;
    MMapFile &operator=(const MMapFile &) = delete;

    // Base address of the mapping (page-aligned by mmap, typically 4 KiB).
    const void *addr() const;
    // Total mapped bytes == file size on disk.
    std::size_t size() const;

private:
    const void *addr_ = nullptr; // mmap result; MAP_FAILED check stored as nullptr on error
    std::size_t size_ = 0;
    // fd_ not stored as a member: close it right after mmap (the mapping holds its own reference to the file, so the mapping outlives the fd per POSIX -- munmap later does not need the fd).
    // Not storing the fd avoids a stale-fd member that could be double-closed.
};
