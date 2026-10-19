#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ─── 하이퍼파라미터 ───────────────────────────────────────
#define IN_C   1
#define OUT_C  8
#define H      32
#define W      32
#define K      3
#define PAD    1
#define STRIDE 1

#define OUT_H  ((H  + 2*PAD - K) / STRIDE + 1)
#define OUT_W  ((W  + 2*PAD - K) / STRIDE + 1)
#define PATCH  (IN_C * K * K)

// ─── im2col (CPU) ─────────────────────────────────────────
void im2col_cpu(const float *X, float *X_col)
{
    for (int c = 0; c < IN_C; c++)
      for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            int row = c*K*K + kh*K + kw;
            for (int oh = 0; oh < OUT_H; oh++)
              for (int ow = 0; ow < OUT_W; ow++) {
                  int col = oh*OUT_W + ow;
                  int ih  = oh*STRIDE - PAD + kh;
                  int iw  = ow*STRIDE - PAD + kw;
                  X_col[row * OUT_H*OUT_W + col] =
                      (ih>=0 && ih<H && iw>=0 && iw<W)
                      ? X[c*H*W + ih*W + iw] : 0.0f;
              }
        }
}

// ─── CPU 기준 convolution ─────────────────────────────────
void conv2d_cpu(const float *X, const float *Wf, float *Y)
{
    float *X_col = (float*)calloc(PATCH * OUT_H*OUT_W, sizeof(float));
    im2col_cpu(X, X_col);

    for (int oc = 0; oc < OUT_C; oc++)
      for (int p = 0; p < OUT_H*OUT_W; p++) {
          float s = 0.0f;
          for (int k = 0; k < PATCH; k++)
              s += Wf[oc*PATCH + k] * X_col[k*OUT_H*OUT_W + p];
          Y[oc*OUT_H*OUT_W + p] = s;
      }
    free(X_col);
}

// ─── main ─────────────────────────────────────────────────
int main(void)
{
    // 호스트 메모리
    int xsz = IN_C  * H * W;
    int wsz = OUT_C * PATCH;
    int ysz = OUT_C * OUT_H * OUT_W;
    int csz = PATCH * OUT_H * OUT_W;

    float *h_X     = (float*)malloc(xsz * sizeof(float));
    float *h_W     = (float*)malloc(wsz * sizeof(float));
    float *h_Y_cpu = (float*)calloc(ysz, sizeof(float));
    float *h_Y_gpu = (float*)calloc(ysz, sizeof(float));
    float *h_Xcol  = (float*)calloc(csz, sizeof(float));

    // 랜덤 초기화 (정수값 -5~5)
    srand(42);
    for (int i = 0; i < xsz; i++) h_X[i] = (float)(rand()%11 - 5);
    for (int i = 0; i < wsz; i++) h_W[i] = (float)(rand()%11 - 5);

    printf("2D Convolution using cuBLAS SGEMM\n");
    printf("Input  : 1 x %d x %d\n", H, W);
    printf("Filter : %d x %d x %d x %d\n", OUT_C, IN_C, K, K);
    printf("Output : %d x %d x %d\n\n", OUT_C, OUT_H, OUT_W);

    // ── CPU 실행 ──────────────────────────────────────────
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    conv2d_cpu(h_X, h_W, h_Y_cpu);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = (t1.tv_sec - t0.tv_sec)*1e3
                  + (t1.tv_nsec - t0.tv_nsec)*1e-6;

    // ── GPU 준비 ──────────────────────────────────────────
    float *d_Xcol, *d_Wcol, *d_Ycol;
    cudaMalloc((void**)&d_Xcol, csz * sizeof(float));
    cudaMalloc((void**)&d_Wcol, wsz * sizeof(float));
    cudaMalloc((void**)&d_Ycol, ysz * sizeof(float));

    // im2col 호스트에서 수행 후 전송
    im2col_cpu(h_X, h_Xcol);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetVector(csz, sizeof(float), h_Xcol, 1, d_Xcol, 1);
    cublasSetVector(wsz, sizeof(float), h_W,    1, d_Wcol, 1);
    cudaMemset(d_Ycol, 0, ysz * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    // ── GPU 실행 (CUDA Event 타이밍) ──────────────────────
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0);

    // Y_col = W_col(OUT_C x PATCH) × X_col(PATCH x OUT_H*OUT_W)
    // cuBLAS column-major 트릭: 
    //   C = A×B  →  C^T = B^T × A^T
    //   row-major Y = row-major W × row-major X
    //   ↓ cuBLAS에 넘길 때 순서 교환
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        OUT_H*OUT_W,   // M
        OUT_C,         // N
        PATCH,         // K
        &alpha,
        d_Xcol, OUT_H*OUT_W,
        d_Wcol, PATCH,
        &beta,
        d_Ycol, OUT_H*OUT_W);

    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev0, ev1);

    // ── 결과 회수 ─────────────────────────────────────────
    cublasGetVector(ysz, sizeof(float), d_Ycol, 1, h_Y_gpu, 1);

    // ── 성능 출력 ─────────────────────────────────────────
    printf("[Performance Result]\n");
    printf("- CPU Time: %f ms\n", cpu_ms);
    printf("- GPU Time: %f ms\n", gpu_ms);
    printf("- Speedup : %fx\n", cpu_ms / gpu_ms);

    // ── 오차 검증 ─────────────────────────────────────────
    double err = 0.0;
    for (int i = 0; i < ysz; i++) {
        double d = h_Y_cpu[i] - h_Y_gpu[i];
        err += d*d;
    }
    printf("- Total Error Norm: %e\n", sqrt(err));
    printf("--------------------------------\n\n");

    // ── 샘플 출력 ─────────────────────────────────────────
    printf("[Sample Output Comparison]\n");
    for (int i = 0; i < 10; i++)
        printf("CPU[%d] = %f, GPU[%d] = %f\n",
               i, h_Y_cpu[i], i, h_Y_gpu[i]);

    // ── 정리 ──────────────────────────────────────────────
    free(h_X); free(h_W); free(h_Y_cpu); free(h_Y_gpu); free(h_Xcol);
    cudaFree(d_Xcol); cudaFree(d_Wcol); cudaFree(d_Ycol);
    cublasDestroy(handle);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    return 0;
}