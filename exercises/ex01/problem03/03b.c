#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Write a line of C code that replaces the LSB of x0 with m using:\n");
    int x0 = 255;
    int m = 0b0;
    printf("x0 = %d, m = 0b%d\n\n", x0, m);

    printf("arithmetic operators: ");
    int x1 = (x0 / 2) * 2 + m;
    printf("x1 = (x0 / 2) * 2 + m = %d\n\n", x1);

    printf("bit masking: ");
    x1 = (x0 & ~0b1) | m;
    printf("x1 = (x0 & ~0b1) | m = %d\n\n", x1);

    printf("bit shifting: ");
    x1 = (x0 >> 1) << 1 | m;
    printf("x1 = (x0 >> 1) << 1 | m = %d\n\n", x1);

    printf("bitwise XOR: ");
    x1 = (x0 & ~0b1) ^ m;
    printf("x1 = (x0 & ~0b1) ^ m = %d\n\n", x1);
    
    x1 = x0 ^ x0 % 2 ^ m;
    printf("x1 = x0 ^ x0 %% 2 ^ m = %d\n\n", x1);

    return EXIT_SUCCESS;
}