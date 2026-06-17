

#include "reference_kernel.h"

#include <cusparse.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/tracer.h"
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_spmv(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, long long n) {
    global_tracer.trace("Entering launch_reference_spmv");
    
    int nnz;
    cudaMemcpy(&nnz, row_offsets + n, sizeof(int), cudaMemcpyDeviceToHost);
    
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, n, n, nnz, 
                      (void*)row_offsets, (void*)col_indices, (void*)values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX;
    cusparseCreateDnVec(&vecX, n, (void*)x, CUDA_R_32F);

    cusparseDnVecDescr_t vecY;
    cusparseCreateDnVec(&vecY, n, (void*)y, CUDA_R_32F);

    float alpha = 1.0f;
    float beta = 0.0f;
    
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, 
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    
    global_tracer.trace("Launching cusparseSpMV");
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, 
                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    cudaFree(dBuffer);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    
    global_tracer.trace("Exiting launch_reference_spmv");
    
    return {{"cusparseSpMV", nullptr, 1, 1, 0}};
}
