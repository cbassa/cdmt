#ifndef __CDMT_h
#define __CDMT_h

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <getopt.h>

#include <cufft.h>

#include "read_hdf5.h"
#include "write_sigproc.h"
#include "error_check.h"

#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10

void usage();


static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b);
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,int l,float scale);

__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nbin,int nchan,int nfft,int nsub,int noverlap,int nsamp,float *fbuf);
__global__ void unpack_and_padd(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
__global__ void swap_spectrum_halves(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny);
__global__ void compute_chirp(double fcen,double bw,float *dm,int nchan,int nbin,int nsub,int ndm,cufftComplex *c);
__global__ void compute_block_sums(float *z,int nchan,int nblock,int nsum,float *bs1,float *bs2);
__global__ void compute_channel_statistics(int nchan,int nblock,int nsum,float *bs1,float *bs2,float *zavg,float *zstd);
__global__ void redigitize(float *z,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz);
__global__ void decimate_and_redigitize(float *z,int ndec,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz);

#ifdef __cplusplus
}
#endif

#endif // __CDMT_h