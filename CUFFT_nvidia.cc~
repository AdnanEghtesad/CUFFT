#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cufft.h>

int main(void) {



#define NX 256
#define BATCH 10
cufftHandle plan;
cufftComplex *data;
cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
/* Create a 1D FFT plan. */
cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
/* Use the CUFFT plan to transform the signal in place. */
cufftExecC2C(plan, data, data, CUFFT_FORWARD);
/* Inverse transform the signal in place. */
cufftExecC2C(plan, data, data, CUFFT_INVERSE);
/* Note:
(1) Divide by number of elements in data set to get back original data
(2) Identical pointers to input and output arrays implies in-place
 transformation
*/
/* Destroy the CUFFT plan. */
cufftDestroy(plan);
cudaFree(data);



return 0;


}
