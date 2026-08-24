#pragma once
#include <unistd.h>

/**
 * RAII guard for a POSIX file descriptor (fd version of std::unique_ptr).
 * Owns exactly one fd; closes it best-effort on destruction (errors swallowed, because the guard is used during stack unwinding and must not throw).
 * Use release() to surrender ownership when you want to manage close yourself.
 */
class UniqueFd
{
public:
    // class 的成员函数如果定义直接写在类体内, 隐式 inline
    // Take ownership of `fd` (a real fd, i.e. >= 0).
    explicit UniqueFd(int fd) noexcept : fd_(fd)
    {
    }

    // If fd_ >= 0: close(fd_), ignore the return value (best-effort, noexcept).
    ~UniqueFd() noexcept
    {
        if (fd_ >= 0)
        {
            close(fd_);
            fd_ = -1;
        }
    }

    // Unique ownership → non-copyable.
    UniqueFd(const UniqueFd &) = delete;
    UniqueFd &operator=(const UniqueFd &) = delete;

    // Move: steal other's fd, leave other holding the "no fd" sentinel (-1).
    UniqueFd(UniqueFd &&other) noexcept : fd_(other.fd_)
    {
        other.fd_ = -1;
    }
    UniqueFd &operator=(UniqueFd &&other) noexcept
    {
        if (this != &other)
        {
            if (fd_ >= 0)
                close(fd_);
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    // Raw fd, or -1 if owning none.
    int get() const noexcept
    {
        return fd_;
    }
    // Hand over ownership: return fd_, set fd_ = -1 so dtor won't close it.
    int release() noexcept
    {
        int fd = fd_;
        fd_ = -1;
        return fd;
    }

private:
    int fd_ = -1; // -1 is the sentinel for "owns no fd" (POSIX valid fds are >= 0)
};
