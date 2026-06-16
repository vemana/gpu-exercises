#ifndef TRACER_CUH
#define TRACER_CUH

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

class Tracer {
private:
    bool quiet;
    std::chrono::time_point<std::chrono::steady_clock> start_time;

public:
    Tracer(bool q = true) : quiet(q) {
        start_time = std::chrono::steady_clock::now();
    }

    void setQuiet(bool q) {
        quiet = q;
    }

    void trace(const std::string& message) {
        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            std::cout << "[" << std::right << std::setw(7) << duration << "] " << message << std::endl;
        }
    }
};

extern Tracer global_tracer;

#endif
