#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#ifdef RISCV_ROCKET
#include <riscv-pk/encoding.h>
#endif

#include "model.h"

#ifdef MLP_MNIST
#include "mlp_test_data.h"
#endif

#ifdef LENET_MNIST
#include "lenet_test_mnist.h"
#endif

#ifdef CIFAR10
#include "test_cifar10.h"
#endif

#ifdef CIFAR100
#include "test_cifar100.h"
#endif

int main(){

    Model model;

    printf("Init Model\n");
    model_init(&model);
    #ifdef REPEAT
    while (1)
    {
    #endif
    
    int correct = 0;

    for (int t=0; t < TEST_NUM; t++){

        printf("Set Input Data\n");
        model.x.data = test_input[t];

        printf("Forward Model\n");
        model_forward(&model);
        MiCo_print_profilers();

        size_t label[1];
        MiCo_argmax2d_f32(label, &model.output);
        printf("Predicted Label: %ld, Correct Label: %d\n", label[0], test_label[t]);
        // MiCo_print_tensor2d_f32(&model.output);
        if (label[0] == test_label[t]){
            correct++;
        }
    }
    printf("Correct: %d / %d\n", correct, TEST_NUM);
    printf("Accuracy: %f\n", (float)correct/TEST_NUM);
    #ifdef REPEAT
    
    }
    #endif
    return 0;
}