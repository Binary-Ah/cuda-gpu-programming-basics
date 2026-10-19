#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void add(long long *a, long long *b, long long *c, int n) {
    int tid = 0;
    while(tid < n) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main(void) {
    printf("N,CPU_Time\n");  // CSV 헤더
    
    for (int n = 10000; n <= 500000; n += 10000) {
        long long *a = (long long *)malloc(n * sizeof(long long));
        long long *b = (long long *)malloc(n * sizeof(long long));
        long long *c = (long long *)malloc(n * sizeof(long long));
        
        for (int i = 0; i < n; i++) {
            a[i] = -i;
            b[i] = (long long)i * i;
        }
        
        clock_t start = clock();
        
        if (n == 500000) {
            int half = n / 2;  // 250000
            add(a, b, c, half);                      // 첫 번째 절반 (0 ~ 249999)
            add(a + half, b + half, c + half, half); // 두 번째 절반 (250000 ~ 499999)
        } else {
            add(a, b, c, n);
        }
        
        clock_t end = clock();
        
        printf("%d,%lf\n", n, (double)(end - start) / CLOCKS_PER_SEC);
        
        free(a);
        free(b);
        free(c);
    }
    return 0;
}
