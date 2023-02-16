#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
private:
    std::chrono::steady_clock::time_point start;

public:
    Timer() : start(std::chrono::steady_clock::now()) {}

    uint64_t ms() {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
};

#endif
