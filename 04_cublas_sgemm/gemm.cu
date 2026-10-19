#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static void simple_sgemm(int n, float alpha, const float *A, const float *B, float beta, float *C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float prod = 0;
            for (int k = 0; k < n; ++k)
                prod += A[k * n + i] * B[j * n + k];
            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

int main(int argc, char **argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    int n2 = N * N;
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasStatus_t status;
    float gpu_time_ms = 0;

    cudaSetDevice(0);
    printf("=== Performance Comparison: CPU vs GPU (Matrix Size: %d x %d) ===\n", N, N);

    cublasCreate(&handle);

    float *h_A = (float*)malloc(n2 * sizeof(float));
    float *h_B = (float*)malloc(n2 * sizeof(float));
    float *h_C = (float*)malloc(n2 * sizeof(float));
    float *h_C_ref = (float*)malloc(n2 * sizeof(float));

    srand(42);
    for (int i = 0; i < n2; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
        h_C[i] = 0.0f;
    }
    memcpy(h_C_ref, h_C, n2 * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n2 * sizeof(float));
    cudaMalloc((void**)&d_B, n2 * sizeof(float));
    cudaMalloc((void**)&d_C, n2 * sizeof(float));

    cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);
    cublasSetVector(n2, sizeof(float), h_B, 1, d_B, 1);
    cublasSetVector(n2, sizeof(float), h_C, 1, d_C, 1);

    printf("Calculating on CPU... ");
    fflush(stdout);
    clock_t cpu_start = clock();
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Done!\n");

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    printf("Calculating on GPU (cuBLAS)... ");
    fflush(stdout);
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                         &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_end);
    printf("Done!\n");

    cublasGetVector(n2, sizeof(float), d_C, 1, h_C, 1);

    printf("\n[Performance Result]\n");
    printf("- CPU Time: %f ms\n", cpu_time_ms);
    printf("- GPU Time: %f ms\n", gpu_time_ms);
    printf("- Speedup:  %fx\n", (float)(cpu_time_ms / gpu_time_ms));
    printf("------------------------------\n");

    float error_norm = 0;
    for (int i = 0; i < n2; ++i) {
        float diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
    }
    printf("Total Error Norm: %e\n\n", sqrt(error_norm));

    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);

    return 0;
}