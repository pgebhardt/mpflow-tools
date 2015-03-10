#ifndef MPFLOW_TOOLS_UTILS_HIGH_PRECISION_TIME_H
#define MPFLOW_TOOLS_UTILS_HIGH_PRECISION_TIME_H

#include <chrono>

class HighPrecisionTime {
private:
    std::chrono::high_resolution_clock::time_point time_;

public:
    HighPrecisionTime() {
        this->restart();
    }
    void restart() {
        this->time_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed() {
        return std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - this->time_).count();
    }
};

#endif
