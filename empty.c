#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "bitserial/driver.h"

int main(){
    
    printf("Hello, World!\n");
    printf("Starting Bitserial Test...\n");

    int busy = mico_bitserial_is_busy();
    printf("Bitserial busy status: %d\n", busy);

    printf("Configuring Bitserial Length to 256...\n");
    mico_bitserial_config(256);

    printf("Starting Bitserial Operation...\n");
    mico_bitserial_start();
    printf("Bitserial Operation Started.\n");
    busy = mico_bitserial_is_busy();
    printf("Bitserial busy status: %d\n", busy);

    printf("Bitserial Test Complete.\n");
    return 0;
}