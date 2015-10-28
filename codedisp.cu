#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10

struct chirp {
  int nbin,nd1,nd2,nd;
  cufftComplex *c;
};

static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s)
{
  cufftComplex c;
  c.x=s*a.x;
  c.y=s*a.y;
  return c;
}

static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b)
{
  cufftComplex c;
  c.x=a.x*b.x-a.y*b.y;
  c.y=a.x*b.y+a.y*b.x;
  return c;
}

// Pointwise complex multiplication (and scaling)
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,float scale)
{
  int i,j,k;
  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  
  if (i<nx && j<ny) {
    k=i+nx*j;
    c[k]=ComplexScale(ComplexMul(a[k],b[i]),scale);
  }
}

// Generate signal (scale Gaussian noise in with a periodic amplitude)
__global__ void generate_signal(cufftComplex *cp1,cufftComplex *cp2,int nsamp,double tsamp,float a0,float a1,float v,float w,float phase)
{
  int i;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;
  float a;
  double t;

  // This loop is a trick to ensure all data is processed regardless of nblock x nthread
  for (i=threadID;i<nsamp;i+=numThreads) {
    // Compute time
    t=(double) i*tsamp;

    // Compute amplitude
    a=a0+a1*exp(w*(cos(2.0*M_PI*(t*v+phase))-1.0));

    // Scale samples with amplitude
    cp1[i].x*=a;
    cp1[i].y*=a;
    cp2[i].x*=a;
    cp2[i].y*=a;
  }

  return;
}

// Convert a float to a byte
__device__ char float2byte(float x)
{
  char c;

  if (x<0.0)
    x+=256.0;

  if (x<0.0)
    x=0.0;
  if (x>255.0)
    x=255.0;

  c=(char) x;

  return c;
}

__global__ void rebit_and_pack(cufftComplex *cp1,cufftComplex *cp2,int nsamp,char *dbuf)
{
  int i;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  // This loop is a trick to ensure all data is processed regardless of nblock x nthread
  for (i=threadID;i<nsamp;i+=numThreads) {
    dbuf[4*i]=float2byte(cp1[i].x);
    dbuf[4*i+1]=float2byte(cp1[i].y);
    dbuf[4*i+2]=float2byte(cp2[i].x);
    dbuf[4*i+3]=float2byte(cp2[i].y);
  }

  return;
}

// Compute chirp
struct chirp get_chirp(double f0,double df,float dm)
{
  int i,j,nexp;
  float tdm;
  float s,rt,t,f,td1,td2;
  struct chirp c;

  // Compute dispersion sweep                                                                       
  td1=dm*(pow(f0,-2)-pow(f0+0.5*df,-2))/DMCONSTANT;
  td2=dm*(pow(f0-0.5*df,-2)-pow(f0,-2))/DMCONSTANT;
  c.nd1=(int) floor(td1*df);
  c.nd2=(int) floor(td2*df);
  tdm=dm*(pow(f0-0.5*df,-2)-pow(f0+0.5*df,-2))/DMCONSTANT;
  c.nd=(int) floor(tdm*df);
  //  c.nd=c.nd1+c.nd2;
  nexp=(int) ceil(log(c.nd)/log(2.0))+1;
  //////////////////// HARDCODED 512k bins //////////////////////////
  /*
  nexp=19;
  c.nd1=16384;
  c.nd2=16384;
  c.nd=c.nd1+c.nd2;
  */
  c.nbin=(int) pow(2.0,nexp);

  s=2.0*M_PI*dm/DMCONSTANT;
  printf("Dispersion sweep: %f us, %d bins\n%d (%d+%d) bins discarded per FFT\n",tdm,c.nbin,c.nd,c.nd1,c.nd2);

  // Allocate
  c.c=(cufftComplex *) malloc(sizeof(cufftComplex)*c.nbin);

  // Compute chirp
  for (i=0;i<c.nbin;i++) {
    if (i<c.nbin/2)
      j=i+c.nbin/2;
    else
      j=i-c.nbin/2;

    f=-0.5*df+df*(float) j/(float) (c.nbin-1);

    rt=f*f*s/((f0+f)*f0*f0);
    t=1.0/sqrt(1.0+pow((f/(0.47*df)),80));

    c.c[i].x=cos(rt)*t;
    c.c[i].y=sin(rt)*t;
  }

  return c;
}

// Overlap-save padding (Fig. 2.5 from Willem van Straten's thesis)
__global__ void padd_data(cufftComplex *y,cufftComplex *x,int n,int nx,int ny,int i0,int m)
{
  int i,j,k,l;

  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  if (i<nx && j<ny) {
    k=i+nx*j;
    l=i+m*j-i0;
    if (l<0 || l>=n) {
      y[k].x=0.0;
      y[k].y=0.0;
    } else {
      y[k].x=x[l].x;
      y[k].y=x[l].y;
    }
  }

  return;
}

// Overlap-save unpadding (Fig. 2.5 from Willem van Straten's thesis)
__global__ void unpadd_data(cufftComplex *y,cufftComplex *x,int n,int nx,int ny,int i0,int i1,int m)
{
  int i,j,k,l;

  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  if (i<nx && j<ny) {
    k=i+nx*j;
    l=i+m*j-i0;
    // Ensure that we select data from unpadded region
    if (l>=0 && l<n && i>=i0 && i<m+i1) {
      y[l].x=x[k].x;
      y[l].y=x[k].y;
    }
  }

  return;
}

int main(int argc, char *argv[])
{
  int nsamp;
  int nx,ny,m;
  float p0=0.5; // phase offset
  float v0=100.0; // spin frequency (Hz)
  float w=1000.0; // von Mises width (k~1/w)
  float a0=2.0,a1=64.0; // amplitude
  float dm=100.0; // DM (pc/cc)
  double f0=135e6; // Frequency (Hz)
  double df=195312.5; // Bandwidth (Hz)
  float length=600.0; // length in (s)
  double tsamp;
  curandGenerator_t genp1,genp2;
  cufftComplex *cp1,*cp2,*cp1p,*cp2p,*dc;
  cufftHandle ftc2cf,ftc2ci;
  char *header,*hbuf,*dbuf;
  FILE *file;
  struct chirp c;
  int idist,odist,iembed,oembed,istride,ostride;
  dim3 blocksize,gridsize;

  // Sampling time
  tsamp=1.0/df;

  // Compute chirp
  c=get_chirp(f0*1e-6,df*1e-6,dm);

  // Samples to compute
  nsamp=(int) floor(length/tsamp);

  // Padding size
  nx=c.nbin;
  m=c.nbin-c.nd;
  ny=(int) ceil(nsamp/(float) m);

  printf("%d samples, %dx%d = %d padded array\n",nsamp,nx,ny,nx*ny);
  printf("%d %d %d %d %d\n",c.nbin,c.nd,c.nd1,c.nd2,m);

  // Set device
  checkCudaErrors(cudaSetDevice(1));

  // Allocate device memory for chirp
  checkCudaErrors(cudaMalloc((void **) &dc,sizeof(cufftComplex)*nx));

  // Copy chirp to device
  checkCudaErrors(cudaMemcpy(dc,c.c,sizeof(cufftComplex)*nx,cudaMemcpyHostToDevice));

  // Set up random number generators
  checkCudaErrors(curandCreateGenerator(&genp1,CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandCreateGenerator(&genp2,CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(genp1,1234ULL));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(genp2,2345ULL));

  // Allocate device memory for dual polarization complex samples
  checkCudaErrors(cudaMalloc((void **) &cp1,sizeof(cufftComplex)*nsamp));
  checkCudaErrors(cudaMalloc((void **) &cp2,sizeof(cufftComplex)*nsamp));

  // Generate random numbers (cufftComplex datatype consists of two floats)
  curandGenerateNormal(genp1,(float *) cp1,2*nsamp,0.0,1.0);
  curandGenerateNormal(genp2,(float *) cp2,2*nsamp,0.0,1.0);
  
  // Impart signal
  generate_signal<<<256,256>>>(cp1,cp2,nsamp,tsamp,a0,a1,v0,w,p0);

  // Allocate device memory for padded dual polarization complex samples
  checkCudaErrors(cudaMalloc((void **) &cp1p,sizeof(cufftComplex)*nx*ny));
  checkCudaErrors(cudaMalloc((void **) &cp2p,sizeof(cufftComplex)*nx*ny));

  // Padd data
  blocksize.x=32;
  blocksize.y=32;
  gridsize.x=nx/blocksize.x+1;
  gridsize.y=ny/blocksize.y+1;
  printf("Grids: %dx%d; Blocks: %dx%d\n",gridsize.x,gridsize.y,blocksize.x,blocksize.y);

  padd_data<<<gridsize,blocksize>>>(cp1p,cp1,nsamp,nx,ny,c.nd1,m);
  padd_data<<<gridsize,blocksize>>>(cp2p,cp2,nsamp,nx,ny,c.nd1,m);

  // Generate FFT plan (batch in-place forward FFT)
  idist=nx;  odist=nx;  iembed=nx;  oembed=nx;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cf,1,&nx,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,ny));

  // Generate FFT plan (batch in-place backward FFT)
  idist=nx;  odist=nx;  iembed=nx;  oembed=nx;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2ci,1,&nx,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,ny));

  // Perform FFT
  checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1p,(cufftComplex *) cp1p,CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2p,(cufftComplex *) cp2p,CUFFT_FORWARD));

  // Perform complex multiplication of FFT'ed data with chirp (in place)
  PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp1p,dc,cp1p,nx,ny,1.0/(float) nx);
  PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp2p,dc,cp2p,nx,ny,1.0/(float) nx);

  // Perform FFT
  checkCudaErrors(cufftExecC2C(ftc2ci,(cufftComplex *) cp1p,(cufftComplex *) cp1p,CUFFT_INVERSE));
  checkCudaErrors(cufftExecC2C(ftc2ci,(cufftComplex *) cp2p,(cufftComplex *) cp2p,CUFFT_INVERSE));

  // Unpadd data
  unpadd_data<<<gridsize,blocksize>>>(cp1,cp1p,nsamp,nx,ny,c.nd1,c.nd2,m);
  unpadd_data<<<gridsize,blocksize>>>(cp2,cp2p,nsamp,nx,ny,c.nd1,c.nd2,m);

  // Allocate memory for redigitized output and header
  header=(char *) malloc(sizeof(char)*HEADERSIZE);
  hbuf=(char *) malloc(sizeof(char)*4*nsamp);
  checkCudaErrors(cudaMalloc((void **) &dbuf,sizeof(char)*4*nsamp));

  // Rebit to 8bits and pack data (interleaving ReP1, ImP1, ReP2, Im P2)
  rebit_and_pack<<<256,256>>>(cp1,cp2,nsamp,dbuf);

  // Copy buffer back to host
  checkCudaErrors(cudaMemcpy(hbuf,dbuf,sizeof(char)*4*nsamp,cudaMemcpyDeviceToHost));

  // Read header
  file=fopen("header.txt","r");
  if (file!=NULL) {
    fread(header,1,HEADERSIZE,file);
    fclose(file);
  } else {
    printf("header.txt not found\n");
  }

  // Write header and buffer
  file=fopen("test.dada","wb");
  fwrite(header,sizeof(char),HEADERSIZE,file);
  fwrite(hbuf,sizeof(char),4*nsamp,file);
  fclose(file);

  // Free host memory
  free(header);
  free(hbuf);
  free(c.c);

  // Free device memory
  cudaFree(cp1);
  cudaFree(cp2);
  cudaFree(cp1p);
  cudaFree(cp2p);
  cudaFree(dbuf);
  cudaFree(dc);

  // Free random number generators
  curandDestroyGenerator(genp1);
  curandDestroyGenerator(genp2);

  // Free plans
  cufftDestroy(ftc2cf);
  cufftDestroy(ftc2ci);

  return 0;
}
