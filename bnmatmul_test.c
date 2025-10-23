#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#include "matmul_test.h"

int main(){
    Tensor2D_Q8 x, w;

    x.data = (int8_t*)malloc(N*K*sizeof(int8_t));
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    int32_t *o = (int32_t*)malloc(N*M*sizeof(int32_t));

    long start_time, end_time;

    start_time = MiCo_time();
    MiCo_Q8x2_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q8x1_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x1 Time: %ld\n", end_time - start_time);

    // Floating Point Kernel
    // float *fx, *fw, *fo;

    // fx = (float*)malloc(N*K*sizeof(float));
    // fw = (float*)malloc(K*M*sizeof(float));
    // fo = (float*)malloc(N*M*sizeof(float));

    // start_time = MiCo_time();
    // MiCo_MatMul_f32(fo, fx, fw, N, K, M);
    // end_time = MiCo_time();

    // printf("Floating Point Time: %ld\n", end_time - start_time);
}