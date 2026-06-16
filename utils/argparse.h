#ifndef ARGPARSE_H
#define ARGPARSE_H

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <algorithm>
#include <stdexcept>

template <size_t D>
struct TestSize {
    std::array<long long, D> dims;

    bool operator<(const TestSize<D>& other) const {
        return dims < other.dims;
    }

    bool dominates(const TestSize<D>& other) const {
        for (size_t i = 0; i < D; ++i) {
            if (dims[i] < other.dims[i]) return false;
        }
        return true;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const TestSize<D>& ts) {
        os << "[";
        for (size_t i = 0; i < D; ++i) {
            os << ts.dims[i] << (i < D - 1 ? ", " : "");
        }
        os << "]";
        return os;
    }
};

enum class FilterMode {
    NONE,
    EXACT_SIZE,
    ABOVE
};

template <size_t D>
struct Config {
    bool verbose = false;
    bool test_ref_kernel_only = false;
    FilterMode mode = FilterMode::NONE;
    TestSize<D> filter_size;
};

template <size_t D>
struct FilterResult {
    std::vector<TestSize<D>> sizes;
    bool fallback_occurred = false;
};

inline std::vector<long long> parse_comma_separated(const std::string& str) {
    std::vector<long long> result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(std::stoll(token));
        } catch (...) {
            std::cerr << "Error: invalid integer in size argument '" << token << "'\n";
            exit(1);
        }
    }
    return result;
}

template <size_t D>
Config<D> parse_args(int argc, char** argv) {
    Config<D> config;
    bool size_specified = false;
    bool above_specified = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -v, --verbose           Enable verbose tracing\n"
                      << "  --size <size>           Run the smallest test_size that is >= <size>\n"
                      << "  --above <size>          Run all test_sizes that are >= <size>\n"
                      << "  --test_ref_kernel_only  Test reference kernel correctness only and exit\n"
                      << "  -h, --help              Show this help message\n\n"
                      << "For multi-dimensional tests, <size> should be a comma-separated list of dimensions (e.g., 1024,2048).\n";
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--test_ref_kernel_only") {
            config.test_ref_kernel_only = true;
        } else if (arg == "--size" || arg == "--above") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires an argument.\n";
                exit(1);
            }
            if (arg == "--size") size_specified = true;
            if (arg == "--above") above_specified = true;
            
            if (size_specified && above_specified) {
                std::cerr << "Error: --size and --above are mutually exclusive. Please specify exactly one.\n";
                exit(1);
            }

            std::string size_str = argv[++i];
            auto vals = parse_comma_separated(size_str);
            if (vals.size() != D) {
                std::cerr << "Error: dimension mismatch. Expected " << D << " dimensions, but got " << vals.size() << " in '" << size_str << "'.\n";
                exit(1);
            }
            for (size_t d = 0; d < D; ++d) {
                config.filter_size.dims[d] = vals[d];
            }
            config.mode = (arg == "--size") ? FilterMode::EXACT_SIZE : FilterMode::ABOVE;
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'\n";
            exit(1);
        }
    }
    return config;
}

template <size_t D>
FilterResult<D> filter_test_sizes(const std::vector<TestSize<D>>& all_sizes, const Config<D>& config) {
    FilterResult<D> result;
    
    std::vector<TestSize<D>> sorted_sizes = all_sizes;
    std::sort(sorted_sizes.begin(), sorted_sizes.end());

    if (config.mode == FilterMode::NONE) {
        result.sizes = sorted_sizes;
        return result;
    }

    std::vector<TestSize<D>> matched;
    for (const auto& ts : sorted_sizes) {
        if (ts.dominates(config.filter_size)) {
            matched.push_back(ts);
        }
    }

    if (matched.empty()) {
        result.fallback_occurred = true;
        result.sizes.push_back(sorted_sizes.back());
    } else {
        if (config.mode == FilterMode::EXACT_SIZE) {
            result.sizes.push_back(matched.front());
        } else if (config.mode == FilterMode::ABOVE) {
            result.sizes = matched;
        }
    }

    return result;
}

#endif
