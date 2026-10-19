# CUDA GPU Programming Basics

GPU 프로그래밍 수업 실습 모음. 커널 실행 구조부터 메모리 전송, cuBLAS를 이용한 행렬곱·합성곱까지 CPU와 직접 비교했다.
아래 수치는 RTX 3090 + CUDA 12.9에서 `make run`으로 다시 측정한 값(`results/`)이다.

| # | 실습 | 내용 | 결과 |
|---|---|---|---|
| 01 | `01_hello` | grid 32 × block 32 = 1,024 스레드의 global thread ID 출력 | 1,024줄 출력 후 `CPU_FIN` (호스트–디바이스 동기화 확인) |
| 02 | `02_vector_add_cpu_vs_gpu` | `long long` 벡터 덧셈, N = 1만~50만, CPU 루프 vs GPU 커널 (N=50만은 커널 2회로 분할) | N=50만: CPU 2.17 ms vs GPU 0.12 ms (커널 시간, 전송 제외) |
| 03 | `03_pinned_memory` | 1,000만 원소 덧셈의 H2D → 커널 → D2H 전체 시간, pageable vs pinned(`cudaMallocHost`) | pageable 43.9 ms → pinned 14.3 ms (약 3배) |
| 04 | `04_cublas_sgemm` | 1024×1024 SGEMM, 단순 3중 루프 CPU vs `cublasSgemm` (warm-up 후 측정) | CPU 3,309 ms vs GPU 0.13 ms, 오차 0 |
| 05 | `05_conv2d_im2col_cublas` | 1×32×32 입력, 8×1×3×3 필터 합성곱을 im2col + `cublasSgemm`으로 변환 | 오차 0. 입력이 작고 첫 cuBLAS 호출(warm-up 없음)을 재서 GPU 49.6 ms가 CPU 0.08 ms보다 느림 — 작은 문제에서는 launch·초기화 오버헤드가 지배 |

## 실행
```bash
make            # bin/ 에 빌드 (기본 -arch=sm_86, 다른 GPU면 make ARCH=sm_75)
make run        # results/ 에 결과 저장
```
기본 `-arch` 없이 빌드하면 RTX 3090에서 커널이 출력 없이 끝나는 경우가 있어 Makefile에서 아키텍처를 명시했다.
`02`의 `book.h` `HANDLE_NULL` 매크로는 줄 이어쓰기(`\`)가 빠져 컴파일되지 않던 것을 고쳤다.
