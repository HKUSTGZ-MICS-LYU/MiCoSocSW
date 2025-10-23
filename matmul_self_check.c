#include "sim_stdlib.h"

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#define N 32
#define M 32
#define K 32

extern void MiCo_Q8_MatMul_MiCo(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

int main(){
    Tensor2D_Q8 x, w;

    x.data = (int8_t*)malloc(N*K*sizeof(int8_t));
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    // Initialize the data with some values
    for (size_t i = 0; i < N * K; i++) {
        x.data[i] = (int8_t)(i % 107); // Example initialization
    }
    for (size_t i = 0; i < K * M; i++) {
        w.data[i] = (int8_t)(176 - (i % 177)); // Example initialization
    }

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    int32_t *o_ref = (int32_t*)malloc(N*M*sizeof(int32_t));
    int32_t *o_cmp = (int32_t*)malloc(N*M*sizeof(int32_t));

    long start, end;
    start = MiCo_time();
    MiCo_Q8_MatMul(o_ref, &x, &w);
    end = MiCo_time();
    long ref_time = end - start;

    start = MiCo_time();
    MiCo_Q8_MatMul_MiCo(o_cmp, &x, &w);
    end = MiCo_time();
    long cmp_time = end - start;

    int32_t error = 0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            if(o_ref[i * M + j] != o_cmp[i * M + j]){
                error++;
                printf("Error at (%d, %d): %d != %d\n", i, j, o_ref[i * M + j], o_cmp[i * M + j]);
            }
        }
    }
    if (error == 0) {
        printf("Self-check passed: No errors found.\n");
        printf("Speedup: %ld/%ld = %f\n", ref_time, cmp_time, (float)ref_time / (float)cmp_time);
        printf("MACs/cycle: %f\n", (float)(N*M*K) / (float)(cmp_time));
    } else {
        printf("Self-check failed: %d errors found.\n", error);
    }
    return 0;
}