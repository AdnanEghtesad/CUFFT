#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cufft.h> 


#define NX 2
#define NY 2
#define LX (2*M_PI)
#define LY (2*M_PI)

#define NUM_POINTS1 1024
#define NUM_POINTS2 1024
#define NUM_POINTS3 1024
#define NUM_POINTS NX*NY //256*256*256 


#define REAL 0
#define IMAG 1


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

/*
    for (int i = 0; i < NUM_POINTS; ++i) {
      float theta = (float)i / (float)(NUM_POINTS) * 2.*M_PI;

#if 0
        vx[i][REAL] = 1.0 * cos(10.0 * theta) +
                          0.5 * cos(25.0 * theta);

        vx[i][IMAG] = 1.0 * sin(10.0 * theta) +
                          0.5 * sin(25.0 * theta);
#else
	vx[i][REAL] = 1.0 * cos(theta);
	vx[i][IMAG] = 0.0;
#endif
    }
*/ 


float *d_vx;
cudaMalloc(&d_vx, NX*NY*sizeof(float));
cudaMemcpy(d_vx, vx, NX*NY*sizeof(float), cudaMemcpyHostToDevice);
cufftHandle planr2c;
cufftHandle planc2r;
cufftPlan2d(&planr2c, NY, NX, CUFFT_R2C);
cufftPlan2d(&planc2r, NY, NX, CUFFT_C2R);
cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_NATIVE);
cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_NATIVE);
cufftExecR2C(planr2c, (cufftReal *)d_vx, (cufftComplex *)d_vx);
cufftExecC2R(planc2r, (cufftComplex *)d_vx, (cufftReal *)d_vx);
cudaMemcpy(vx, d_vx, NX*NY*sizeof(cufftReal), cudaMemcpyDeviceToHost);
for (int j = 0; j < NY; j++){
    for (int i = 0; i < NX; i++){
        printf("%.3f ", vx[j*NX + i]/(NX*NY));
    }
    printf("\n");
} 

return 0;

}