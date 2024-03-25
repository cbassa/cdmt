#ifndef __ERROR_CHECK_h
#define __ERROR_CHECK_h

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#include <cufft.h>
#include <cuda_runtime.h>

void __checkCudaErrors(cudaError_t code, const char *file, int line);
void __checkCufftErrors(cufftResult code, const char *file, int line);

#define checkCudaErrors(ans) {__checkCudaErrors((ans), __FILE__, __LINE__);}
#define checkCufftErrors(ans) {__checkCufftErrors((ans), __FILE__, __LINE__);}

#ifdef __cplusplus
}
#endif

#endif // __ERROR_CHECK_h