

#include "reference_kernel.h"

#include <cublas_v2.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/tracer.h"
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_gemm(const float* a, const float* b, float* c, long long n) {
    global_tracer.trace("Entering launch_reference_gemm");
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    global_tracer.trace("Launching cublasSgemm");
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, 
                b, n, 
                a, n, 
                &beta, 
                c, n);
                
    cublasDestroy(handle);
    
    global_tracer.trace("Exiting launch_reference_gemm");
    
    return {{"cublasSgemm", nullptr, 1, 1, 0}};
}
