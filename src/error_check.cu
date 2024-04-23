#include "error_check.h"

void __checkCudaErrors(cudaError_t code, const char *file, int line)
{
    /* Wrapper function for GPU/CUDA error handling. Every CUDA call goes through
       this function. It will return a message giving you the error string,
       file name and line of the error. Aborts on error. */

    if (code != 0)
    {
        fprintf(stderr, "CUDA Error :: %s - %s (line %d)\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void __checkCufftErrors(cufftResult code, const char *file, int line)
{
    /* Wrapper function for CUFFT error handling. Every CUDA call goes through
       this function. It will return a message giving you the error string,
       file name and line of the error. Aborts on error. */

    if (code != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT Error :: Failed with error code %d\n", code);
        exit(EXIT_FAILURE);
    }
}