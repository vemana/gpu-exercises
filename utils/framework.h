#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>
#include "argparse.h"
#include "utils.h"
#include "tracer.h"

extern Tracer global_tracer;

template <size_t D>
struct TestSummary {
    TestSize<D> size;
    OccupancyMetrics student_metrics;
    float speedup;
    float student_ms;
    float ref_ms;
};

template <size_t D>
std::string format_size_commas(const TestSize<D>& size) {
    if (D == 1) return format_with_commas(size.dims[0]);
    std::stringstream ss;
    for (size_t i = 0; i < D; ++i) {
        ss << format_with_commas(size.dims[i]);
        if (i < D - 1) ss << "x";
    }
    return ss.str();
}

template <size_t D>
struct ProblemTest {
    TestSize<D> size;
    ProblemTest(const TestSize<D>& size) : size(size) {}
    virtual ~ProblemTest() = default;

    virtual void generate_test_data(bool check) = 0;

    virtual void setup_reference() {}
    virtual LaunchMetrics launch_reference() = 0;
    virtual void teardown_reference() {}

    virtual void setup_student() {}
    virtual LaunchMetrics launch_student() = 0;
    virtual void teardown_student() {}

    virtual CorrectnessResult verify() = 0;
    virtual void print_mismatch() = 0;
    
    virtual double get_bandwidth_bytes() = 0;
    virtual void teardown() = 0;
};

template <size_t D>
void print_test_header(const TestSize<D>& ts) {
    std::string header_text = "TEST SIZE: " + format_size_commas(ts);
    int padding = std::max(0, (80 - (int)header_text.length()) / 2);
    std::string padded_text = std::string(padding, ' ') + header_text;

    std::cout << "\n\033[1;32m" << std::string(80, '=') << "\n";
    std::cout << padded_text << "\n";
    std::cout << std::string(80, '=') << "\033[0m\n";
}

template <size_t D, typename TestImpl>
void run_correctness_tests(const Config<D>& config, const std::vector<TestSize<D>>& correctness_sizes) {
    for (const auto& ts : correctness_sizes) {
        global_tracer.trace("Starting test_size for size " + format_size_commas(ts) + " (check=1)");
        print_test_header(ts);

        TestImpl test(ts);
        test.generate_test_data(true);

        global_tracer.trace("Testing Reference Kernel Correctness...");
        test.setup_reference();
        LaunchMetrics ref_metrics = test.launch_reference();
        cudaDeviceSynchronize();
        print_occupancy("Reference Kernel", ref_metrics.metrics, format_size_commas(ts), ref_metrics.blocksPerGrid);
        
        std::cout << "[Size " << std::right << std::setw(15) << format_size_commas(ts) << "] (Reference Kernel) ";
        CorrectnessResult res = test.verify();
        print_correctness_result(res);
        if (!res.passed && ts.dims[0] <= 2048) {
            test.print_mismatch();
        }
        test.teardown_reference();

        if (config.test_ref_kernel_only) {
            test.teardown();
            continue;
        }

        global_tracer.trace("Testing Student Kernel Correctness...");
        test.setup_student();
        LaunchMetrics student_metrics = test.launch_student();
        cudaDeviceSynchronize();
        print_occupancy("Student Kernel", student_metrics.metrics, format_size_commas(ts), student_metrics.blocksPerGrid);
        
        std::cout << "[Size " << std::right << std::setw(15) << format_size_commas(ts) << "] (Student Kernel) ";
        res = test.verify();
        print_correctness_result(res);
        if (!res.passed && ts.dims[0] <= 2048) {
            test.print_mismatch();
        }
        test.teardown_student();
        test.teardown();
    }
}

template <size_t D, typename TestImpl>
void run_performance_tests(const Config<D>& config, const std::vector<TestSize<D>>& perf_sizes, std::vector<TestSummary<D>>& summaries) {
    auto filter_result = filter_test_sizes(perf_sizes, config);
    if (filter_result.fallback_occurred) {
        global_tracer.trace("WARNING: No predefined test sizes matched the filter. Falling back to the largest available size.");
    }

    for (const auto& ts : filter_result.sizes) {
        global_tracer.trace("Starting test_size for size " + format_size_commas(ts) + " (check=0)");
        print_test_header(ts);

        TestImpl test(ts);
        test.generate_test_data(false);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float ref_ms = 0.0f;
        float student_ms = 0.0f;

        global_tracer.trace("Recording Reference Kernel Performance...");
        test.setup_reference();
        cudaEventRecord(start);
        LaunchMetrics ref_metrics = test.launch_reference();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ref_ms, start, stop);
        test.teardown_reference();
        print_occupancy("Reference Kernel", ref_metrics.metrics, format_size_commas(ts), ref_metrics.blocksPerGrid);

        global_tracer.trace("Recording Student Kernel Performance...");
        test.setup_student();
        cudaEventRecord(start);
        LaunchMetrics student_metrics = test.launch_student();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&student_ms, start, stop);
        test.teardown_student();
        print_occupancy("Student Kernel", student_metrics.metrics, format_size_commas(ts), student_metrics.blocksPerGrid);

        double bytes = test.get_bandwidth_bytes();
        double ref_bw = (bytes / (ref_ms / 1000.0)) / 1e9;
        double student_bw = (bytes / (student_ms / 1000.0)) / 1e9;
        float speedup = ref_ms / student_ms;

        std::cout << "[Size " << std::right << std::setw(15) << format_size_commas(ts) << "] Performance:" << std::endl;
        std::cout << "  Reference Kernel : " << std::fixed << std::setprecision(2) << ref_ms << " ms (" << ref_bw << " GB/s)" << std::endl;
        std::cout << "  Student Kernel   : " << std::fixed << std::setprecision(2) << student_ms << " ms (" << student_bw << " GB/s)" << std::endl;
        std::cout << "  Speedup (Student/Teacher) : " << std::fixed << std::setprecision(2) << speedup << "x\n" << std::endl;

        summaries.push_back({ts, student_metrics.metrics, speedup, student_ms, ref_ms});
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        test.teardown();
    }
}

template <size_t D>
void print_summary_table(const std::vector<TestSummary<D>>& summaries) {
    std::string header_text = "Final Utilization & Speedup Summary";
    int padding = std::max(0, (105 - (int)header_text.length()) / 2);
    std::string padded_text = std::string(padding, ' ') + header_text;

    std::cout << "\n\033[1;32m" << std::string(105, '=') << "\n";
    std::cout << padded_text << "\n";
    std::cout << std::string(105, '=') << "\033[0m\n";
    std::cout << std::left << std::setw(20) << "Problem Size"
              << std::right << std::setw(15) << "Block Util"
              << std::setw(15) << "Thread Util"
              << std::setw(15) << "SMem Util"
              << std::setw(15) << "Reg Util"
              << std::setw(25) << "Speedup" << std::endl;
    std::cout << std::string(105, '-') << std::endl;
    for (const auto& s : summaries) {
        char buf[64];
        std::cout << std::left << std::setw(20) << format_size_commas(s.size);
        
        snprintf(buf, sizeof(buf), "%.2f%%", s.student_metrics.block_util);
        std::cout << std::right << std::setw(15) << buf;
        
        snprintf(buf, sizeof(buf), "%.2f%%", s.student_metrics.thread_util);
        std::cout << std::setw(15) << buf;
        
        snprintf(buf, sizeof(buf), "%.2f%%", s.student_metrics.smem_util);
        std::cout << std::setw(15) << buf;
        
        snprintf(buf, sizeof(buf), "%.2f%%", s.student_metrics.reg_util);
        std::cout << std::setw(15) << buf;
        
        snprintf(buf, sizeof(buf), "%.2fx (%.2f/%.2f ms)", s.speedup, s.student_ms, s.ref_ms);
        std::cout << std::setw(25) << buf << std::endl;
    }
    std::cout << std::endl;
}

template <size_t D, typename TestImpl>
void run_test_suite(const std::string& name, const Config<D>& config,
                    const std::vector<TestSize<D>>& correctness_sizes,
                    const std::vector<TestSize<D>>& perf_sizes) {
    global_tracer.setQuiet(!config.verbose);
    global_tracer.trace("Program started.");
    
    std::cout << "=== " << name << " ===" << std::endl;

    if (!config.skip_correctness_tests) {
        run_correctness_tests<D, TestImpl>(config, correctness_sizes);

        if (config.test_ref_kernel_only) {
            std::cout << "Reference kernel correctness tested. Exiting early due to --test_ref_kernel_only flag." << std::endl;
            return;
        }
    } else if (config.test_ref_kernel_only) {
        std::cout << "Warning: --skip_correctness_tests and --test_ref_kernel_only are mutually exclusive. Exiting." << std::endl;
        return;
    }

    std::vector<TestSummary<D>> summaries;
    run_performance_tests<D, TestImpl>(config, perf_sizes, summaries);
    print_summary_table<D>(summaries);

    global_tracer.trace("Program finished.");
}

#endif
