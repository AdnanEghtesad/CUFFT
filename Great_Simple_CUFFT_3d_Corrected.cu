#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>
#include <cufft.h> 


#define NX 512
#define NY 512
#define NZ 512
#define LX (2*M_PI)
#define LY (2*M_PI)
#define LZ (2*M_PI)



int main() {
  
 double acc_time;
 int acc_n;
 
float *x = new float[NX*NY*NZ];
float *y = new float[NX*NY*NZ];
float *z = new float[NX*NY*NZ];
float *vx = new float[NX*NY*NZ];

 
 for(int k = 0; k < NZ; k++){
  for(int j = 0; j < NY; j++){
    for(int i = 0; i < NX; i++){
    
        x[j*NX + i] = i * LX/NX;
        y[j*NX + i] = j * LY/NY;
	z[j*NX + i] = k * LZ/NZ;
        vx[j*NX + i] = cos(x[j*NX + i]);
	}
    }
}
float *d_vx;
cudaMalloc(&d_vx, NX*NY*NZ*sizeof(cufftComplex));
cudaMemcpy(d_vx, vx, NX*NY*NZ*sizeof(cufftComplex), cudaMemcpyHostToDevice);
cufftHandle planr2c;
//cufftHandle planc2r;
cufftPlan3d(&planr2c, NZ,NY, NX, CUFFT_R2C);
//cufftPlan3d(&planc2r, NZ,NY, NX, CUFFT_C2R);
cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_NATIVE);
//cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_NATIVE);

    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

cufftExecR2C(planr2c, (cufftReal *)d_vx, (cufftComplex *)d_vx);
//cufftExecC2R(planc2r, (cufftComplex *)d_vx, (cufftReal *)d_vx);
cudaMemcpy(vx, d_vx, NX*NY*NZ*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
   
    clock_gettime(CLOCK_REALTIME, &now);
    acc_time += (now.tv_sec+now.tv_nsec*1e-9) - (tmstart.tv_sec+tmstart.tv_nsec*1e-9);
    acc_n++;

      

    printf("avg CUFFT time : %g total time %g\n", acc_time / acc_n, acc_time);
    getchar();


 for(int k = 0; k < NZ; k++){
  for (int j = 0; j < NY; j++){
    for (int i = 0; i < NX; i++){
        printf("%.3f ", vx[j*NX + i]/ (NX*NY*NZ));
    }
    printf("\n");
  } 
 }
return 0;

}