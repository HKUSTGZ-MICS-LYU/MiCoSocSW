#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#define N 4
#define K 256
#define M 32

qbyte x_array[N*K];
qbyte w_array[K*M];

int main(){
    
    Tensor2D_Q8 x, w;
    printf("Hello MiCo!\n");

    x.data = x_array;
    w.data = w_array;

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    for (size_t i = 0; i < N*K; i++) {
        x.data[i] = (i % 8);
    }
    for (size_t i = 0; i < K*M; i++) {
        w.data[i] = (i + 7 % 8);
    }
    #ifdef REPEAT
    while(1){
    #endif 
        int32_t *o = (int32_t*)malloc(N*M*sizeof(int32_t));
        long start_time, end_time;
        printf("MiCo 8x8 MatMul Test\n");
        start_time = MiCo_time();
        MiCo_Q8_MatMul(o, &x, &w);
        end_time = MiCo_time();
        printf("MiCo 8x8 Time: %ld\n", end_time - start_time);
        free(o);
    #ifdef REPEAT
        delay(1000); // 1 second
    }
    #endif

    return 0;
}