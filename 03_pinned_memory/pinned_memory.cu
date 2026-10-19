#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void run_test(bool use_pinned, int N) {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (use_pinned) {
        cudaMallocHost((void **)&h_a, size);
        cudaMallocHost((void **)&h_b, size);
        cudaMallocHost((void **)&h_c, size);
    } else {
        h_a = (int *)malloc(size);
        h_b = (int *)malloc(size);
        h_c = (int *)malloc(size);
    }


    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i ;
    }
    cudaEventRecord(start);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s memory end-to-end 시간: %f ms\n", use_pinned ? "pinned" : "pageable", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    if (use_pinned) {
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
    } else {
        free(h_a);
        free(h_b);
        free(h_c);
    }
}

int main() {
    int N = 10000000; // 1M elements
    run_test(false, N);
    run_test(true, N);

    return 0;
}