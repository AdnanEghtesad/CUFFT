#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>

#define NX 32
#define NY 32
#define LX (2*M_PI)
#define LY (2*M_PI)
int main(void) {



float *x = new float[NX*NY];
float *y = new float[NX*NY];
float *vx = new float[NX*NY];
for(int j = 0; j < NY; j++){
    for(int i = 0; i < NX; i++){
        x[j*NX + i] = i * LX/NX;
        y[j*NX + i] = j * LY/NY;
        vx[j*NX + i] = cos(x[j*NX + i]);
    }
}
float *d_vx;
CUDA_CHECK(cudaMalloc(&d_vx, NX*NY*sizeof(cufftComplex)));
CUDA_CHECK(cudaMemcpy(d_vx, vx, NX*NY*sizeof(cufftComplex), cudaMemcpyHostToDevice));
cufftHandle planr2c;
cufftHandle planc2r;
CUFFT_CHECK(cufftPlan2d(&planr2c, NY, NX, CUFFT_R2C));
CUFFT_CHECK(cufftPlan2d(&planc2r, NY, NX, CUFFT_C2R));
CUFFT_CHECK(cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_NATIVE));
CUFFT_CHECK(cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_NATIVE));
CUFFT_CHECK(cufftExecR2C(planr2c, (cufftReal *)d_vx, d_vx));
CUFFT_CHECK(cufftExecC2R(planc2r, d_vx, (cufftReal *)d_vx));
CUDA_CHECK(cudaMemcpy(vx, d_vx, NX*NY*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
for (int j = 0; j < NY; j++){
    for (int i = 0; i < NX; i++){
        printf("%.3f ", vx[j*NX + i]/(NX*NY));
    }
    printf("\n");
}

}
