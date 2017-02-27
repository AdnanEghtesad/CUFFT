#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cufft.h>


#define NX 200

#define NY 100

int main(void) {

cufftHandle plan;

cufftComplex *data1, *data2;

cudaMalloc((void**)&data1, sizeof(cufftComplex)*NX*NY);

cudaMalloc((void**)&data2, sizeof(cufftComplex)*NX*NY);

/* Create a 2D FFT plan. */

cufftPlan2d(&plan, NX, NY, CUFFT_DATA_C2C);

/* Use the CUFFT plan to transform the signal out of place.

*/

cufftExecute(plan, data1, data2, CUFFT_FORWARD);

}
