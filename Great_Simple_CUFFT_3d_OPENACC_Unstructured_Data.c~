#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
//#include <cuda_runtime.h>
#include <cufft.h> 

#define NX 2
#define NY 2
#define NZ 2
#define LX (2*M_PI)
#define LY (2*M_PI)
#define LZ (2*M_PI)

typedef float fftw_complex[2];

#define REAL 0
#define IMAG 1

int main()
{
  double acc_time = 0;
  int acc_n = 0;
  
  
    /*  FOR C++  */	         
  //fftw_complex *vx = new fftw_complex[NX*NY*NZ];
   /*  FOR  C  */
  fftw_complex*vx = malloc( (NX*NY*NZ)*sizeof(fftw_complex));
  
   #pragma acc enter data create(vx[0:NX*NY*NZ][0:1])
  //{
 
  
  #pragma acc kernels 
  
  {
  
  for(int k = 0; k < NZ; k++){
    for(int j = 0; j < NY; j++){
      for(int i = 0; i < NX; i++){
	float x = i * LX/NX;
	vx[(k*NY+j)*NX + i][REAL] = cos(x);
	vx[(k*NY+j)*NX + i][IMAG] = 0.;
      }
    }
  }
  
  } //  #pragma acc end kernels

 
 
 
  //float *d_vx;
  
  struct timespec now, tmstart;
  clock_gettime(CLOCK_REALTIME, &tmstart);
  
  
  //cudaMalloc(&d_vx, NX*NY*NZ*sizeof(fftw_complex));
 
  
  //cudaMemcpy(d_vx, vx, NX*NY*NZ*sizeof(fftw_complex), cudaMemcpyHostToDevice);
  
  
  
  cufftHandle planc2c;
  cufftPlan3d(&planc2c, NZ,NY, NX, CUFFT_C2C);
  cufftSetCompatibilityMode(planc2c, CUFFT_COMPATIBILITY_NATIVE);

  
  
 // cufftExecC2C(planc2c, (cufftComplex *)d_vx, (cufftComplex *)d_vx, CUFFT_FORWARD);
 
  #pragma acc host_data use_device(vx)
  {

    cufftExecC2C(planc2c, (cufftComplex *)vx, (cufftComplex *)vx, CUFFT_FORWARD);

  } //#pragma acc host_data use_device(vx)
  
  
  
  //cudaMemcpy(vx, d_vx, NX*NY*NZ*sizeof(fftw_complex), cudaMemcpyDeviceToHost);
  
  
  clock_gettime(CLOCK_REALTIME, &now);
  acc_time += (now.tv_sec+now.tv_nsec*1e-9) - (tmstart.tv_sec+tmstart.tv_nsec*1e-9);
  acc_n++;
  printf("avg CUFFT time : %g total time %g\n", acc_time / acc_n, acc_time);
 
#if 1									       
  getchar();
#endif


 #pragma acc exit data copyout(vx[0:NX*NY*NZ][0:1])
  //} //ACC data create(vx)

  for(int k = 0; k < NZ; k++){
    for (int j = 0; j < NY; j++){
      for (int i = 0; i < NX; i++){
	printf("(%.3f,%.3f) ",
	       vx[(k*NY + j)*NX + i][REAL]  ,
	       vx[(k*NY + j)*NX + i][IMAG] );
      }
      printf("\n");
    } 
    printf("\n");
  }
  

  
  return 0;

}
 
