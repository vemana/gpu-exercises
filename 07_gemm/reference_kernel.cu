#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/utils.h"
#include "../utils/tracer.h"

LaunchMetrics launch_reference_gemm(const float* a, const float* b, float* c, int n) {
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
    
    OccupancyMetrics occ;
    occ.is_dummy = true;
    return {1, occ};
}
