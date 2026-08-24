#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "mmap_file.h"
#include "unique_fd.h"

// explicit 只能在声明(.h)出现一次, 定义(.cpp)不能重复写
MMapFile::MMapFile(const std::string &path)
{
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("open file fail");
    }
    UniqueFd guard_fd(fd);

    struct stat fsta;
    if (fstat(fd, &fsta) == -1)
    {
        throw std::runtime_error("get file attributes fail");
    }
    size_ = fsta.st_size;

    // 空文件 mmap 行为未定义
    if (size_ == 0)
    {
        throw std::runtime_error("empty file");
    }
    // 多 worker 共享物理页
    addr_ = mmap(0, size_, PROT_READ, MAP_SHARED, fd, 0);
    if (addr_ == MAP_FAILED)
    {
        addr_ = nullptr;
        throw std::runtime_error("mmap file fail");
    }
    fd = guard_fd.release();
    if (close(fd) == -1)
    {
        munmap(const_cast<void *>(addr_), size_);
        throw std::runtime_error("close file fail");
    }
}

MMapFile::~MMapFile()
{
    if (addr_)
    {
        // 析构函数绝不能抛异常
        // 如果这个对象恰好在栈展开(另一个异常正在往上抛)过程中被析构, 析构再抛第二个异常 → C++ 运行时直接 std::terminate, 进程挂。 没有任何恢复机会
        // 即使正常析构, 抛异常也会让"对象已销毁"这个基本假设破裂, 调用方没法处理
        // 失败只能记日志/忽略, 不能抛
        munmap(const_cast<void *>(addr_), size_);
    }
}

const void *MMapFile::addr() const
{
    return addr_;
}

std::size_t MMapFile::size() const
{
    return size_;
}