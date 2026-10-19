#include "book.h"

__global__ void add(long long *a, long long *b, long long *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    printf("N,GPU_Time\n");  // CSV 헤더
    
    for (int n = 10000; n <= 500000; n += 10000) {
        long long *a = (long long *)malloc(n * sizeof(long long));
        long long *b = (long long *)malloc(n * sizeof(long long));
        long long *c = (long long *)malloc(n * sizeof(long long));
        long long *dev_a, *dev_b, *dev_c;

        HANDLE_ERROR(cudaMalloc((void**)&dev_a, n * sizeof(long long)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_b, n * sizeof(long long)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_c, n * sizeof(long long)));

        for (int i = 0; i < n; i++) {
            a[i] = -i;
            b[i] = (long long)i * i;
        }

        HANDLE_ERROR(cudaMemcpy(dev_a, a, n * sizeof(long long), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_b, b, n * sizeof(long long), cudaMemcpyHostToDevice));
        
        int threadsPerBlock = 256;
        
        clock_t start = clock();
        
        if (n == 500000) {
            // N=500000인 경우 N/2로 나누어 두 번 커널 수행
            int half = n / 2;
            int blocksPerGrid = (half + threadsPerBlock - 1) / threadsPerBlock;
            
            // 첫 번째 절반 처리 (0 ~ half-1)
            add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, half);
            
            // 두 번째 절반 처리 (half ~ n-1)
            add<<<blocksPerGrid, threadsPerBlock>>>(dev_a + half, dev_b + half, dev_c + half, half);
        } else {
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, n);
        }
        
        cudaDeviceSynchronize();  // GPU 완료 대기
        clock_t end = clock();
        
        printf("%d,%lf\n", n, (double)(end - start) / CLOCKS_PER_SEC);

        HANDLE_ERROR(cudaFree(dev_a));
        HANDLE_ERROR(cudaFree(dev_b));
        HANDLE_ERROR(cudaFree(dev_c));
        free(a);
        free(b);
        free(c);
    }
    return 0;
}