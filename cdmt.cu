#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10

// Compute chirp                                                                
void get_channel_chirp(double f0,double df,float dm,int nchan,int nbin,cufftComplex *c)
{
  int i,k,l,m;
  float s,rt,t,f,fc0,dfc;

  // Main constant
  s=2.0*M_PI*dm/DMCONSTANT;

  // Number of channels per subband
  m=nbin/nchan;

  dfc=df/nchan;

  // Loop over subbands
  for (k=0;k<nchan;k++) {
    fc0=f0-0.5*df+df*(float) k/(float) nchan+0.5*df/(float) nchan;
    for (i=0;i<m;i++) {
      f=-0.5*dfc+dfc*(float) i/(float) (m-1);

      rt=-f*f*s/((fc0+f)*fc0*fc0);
      t=1.0/sqrt(1.0+pow((f/(0.47*dfc)),80));

      l=i+k*m;
      
      c[l].x=cos(rt)*t;
      c[l].y=sin(rt)*t;
    }
  }

  return;
}

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

__global__ void unpack_and_padd(char *dbuf,int n,int nx,int ny,int i0,int m,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t i,j,k,l;

  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  if (i<nx && j<ny) {
    k=i+nx*j;
    l=i+m*j-i0;
    if (l<0 || l>=n) {
      cp1[k].x=0.0;
      cp1[k].y=0.0;
      cp2[k].x=0.0;
      cp2[k].y=0.0;
    } else {
      cp1[k].x=(float) dbuf[4*l];
      cp1[k].y=(float) dbuf[4*l+1];
      cp2[k].x=(float) dbuf[4*l+2];
      cp2[k].y=(float) dbuf[4*l+3];
    }
  }

  return;
}

__global__ void swap_spectrum_halves(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny)
{
  int64_t i,j,k,l,m;
  cufftComplex tp1,tp2;

  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  if (i<nx/2 && j<ny) {
    if (i<nx/2)
      k=i+nx/2;
    else
      k=i-nx/2;
    l=i+nx*j;
    m=k+nx*j;
    tp1.x=cp1[l].x;
    tp1.y=cp1[l].y;
    tp2.x=cp2[l].x;
    tp2.y=cp2[l].y;
    cp1[l].x=cp1[m].x;
    cp1[l].y=cp1[m].y;
    cp2[l].x=cp2[m].x;
    cp2[l].y=cp2[m].y;
    cp1[m].x=tp1.x;
    cp1[m].y=tp1.y;
    cp2[m].x=tp2.x;
    cp2[m].y=tp2.y;
  }

  return;
}

__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny,int nz,int i0,int i1,int n,float *fbuf)
{
  int64_t i,j,k,l,m,ii,jj;
  
  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;
  k=blockIdx.z*blockDim.z+threadIdx.z;
  if (i<nx && j<ny && k<nz) {
    m=i+nx*(j+ny*k);

    // Time index
    ii=i+(nx-i0-i1)*k-i0;

    // Frequency index
    jj=ny-j-1;

    // Array index
    l=jj+ny*ii;

    if (i>=i0 && i<=nx-i1 && ii>=0 && ii<n)
      fbuf[l]=sqrt(cp1[m].x*cp1[m].x+cp1[m].y*cp1[m].y+cp2[m].x*cp2[m].x+cp2[m].y*cp2[m].y);
  }

  return;
}

int main(int argc,char *argv[])
{
  int nsamp,nz,mx,my,mz,m,nchan=8,nbin=65536,noverlap=2048;
  int iblock,nread;
  char *header,*hbuf,*dbuf;
  FILE *file,*ofile;
  float *fbuf,*dfbuf;
  cufftComplex *cp1,*cp2,*dc,*c;
  cufftHandle ftc2cf,ftc2cb;
  int idist,odist,iembed,oembed,istride,ostride;
  dim3 blocksize,gridsize;

  c=(cufftComplex *) malloc(sizeof(cufftComplex)*nbin);

  // Compute chirp
  //get_channel_chirp(119.7265625,0.1953125,39.659298,nchan,nbin,c);
  get_channel_chirp(135.0,0.1953125,100.0,nchan,nbin,c);

  nsamp=195312.5;
  nsamp*=600;
  nsamp=491520*20;

  // Data size
  m=nbin-2*noverlap;
  nz=(int) ceil(nsamp/(float) m);
  my=nchan;
  mx=nbin/my;
  mz=nz;
  printf("%dx%d %dx%dx%d %d\n",nbin,nz,mx,my,mz,m);

  // Allocate memory for complex timeseries
  checkCudaErrors(cudaMalloc((void **) &cp1,sizeof(cufftComplex)*nbin*nz));
  checkCudaErrors(cudaMalloc((void **) &cp2,sizeof(cufftComplex)*nbin*nz));

  // Allocate device memory for chirp                                                                
  checkCudaErrors(cudaMalloc((void **) &dc,sizeof(cufftComplex)*nbin));

  // Allocate memory for redigitized output and header
  header=(char *) malloc(sizeof(char)*HEADERSIZE);
  hbuf=(char *) malloc(sizeof(char)*4*nsamp);
  checkCudaErrors(cudaMalloc((void **) &dbuf,sizeof(char)*4*nsamp));

  // Allocate output buffers
  fbuf=(float *) malloc(sizeof(float)*nsamp);
  checkCudaErrors(cudaMalloc((void **) &dfbuf,sizeof(float)*nsamp));

  // Generate FFT plan (batch in-place forward FFT)
  idist=nbin;  odist=nbin;  iembed=nbin;  oembed=nbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cf,1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nz));

  // Generate FFT plan (batch in-place backward FFT)
  idist=mx;  odist=mx;  iembed=mx;  oembed=mx;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cb,1,&mx,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,my*mz));

  // Copy chirp to device                                                                            
  checkCudaErrors(cudaMemcpy(dc,c,sizeof(cufftComplex)*nbin,cudaMemcpyHostToDevice));

  // Read fil file header and dump in output file
  file=fopen("header.fil","r");
  fread(header,sizeof(char),351,file);
  fclose(file);
  ofile=fopen("test.fil","w");
  fwrite(header,sizeof(char),351,ofile);

  // Read file and buffer
  file=fopen("test.dada","r");
  fread(header,sizeof(char),HEADERSIZE,file);

  // Loop over input file contents
  for (iblock=0;;iblock++) {
    nread=fread(hbuf,sizeof(char),4*nsamp,file)/4;
    printf("%d %d %d\n",iblock,nread,4*nsamp);
    if (nread==0)
      break;

    // Copy buffer to device
    checkCudaErrors(cudaMemcpy(dbuf,hbuf,sizeof(char)*4*nread,cudaMemcpyHostToDevice));

    // Unpack data and padd data
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1;
    gridsize.y=nz/blocksize.y+1;
    gridsize.z=1;
    unpack_and_padd<<<gridsize,blocksize>>>(dbuf,nread,nbin,nz,noverlap,m,cp1,cp2);
    
    // Perform FFTs
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_FORWARD));
    
    // Swap spectrum halves for large FFTs
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=mx*my/blocksize.x+1;
    gridsize.y=mz/blocksize.y+1;
    gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,mx*my,mz);
    
    // Perform complex multiplication of FFT'ed data with chirp (in place)                             
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1;
    gridsize.y=nz/blocksize.y+1;
    gridsize.z=1;
    PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp1,dc,cp1,nbin,nz,1.0/(float) nbin);
    PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp2,dc,cp2,nbin,nz,1.0/(float) nbin);
    
    // Swap spectrum halves for small FFTs
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=mx/blocksize.x+1;
    gridsize.y=my*mz/blocksize.y+1;
    gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,mx,my*mz);
    
    // Perform FFTs
    checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_INVERSE));
    checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_INVERSE));
    
    // Detect data
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=mx/blocksize.x+1;
    gridsize.y=my/blocksize.y+1;
    gridsize.z=mz/blocksize.z+1;
    transpose_unpadd_and_detect<<<gridsize,blocksize>>>(cp1,cp2,mx,my,mz,noverlap/my,noverlap/my,nread/my,dfbuf);
    
    // Copy buffer to host
    checkCudaErrors(cudaMemcpy(fbuf,dfbuf,sizeof(float)*nread,cudaMemcpyDeviceToHost));
    
    // Write buffer
    fwrite(fbuf,sizeof(float),nread,ofile);

  }

  // Close files
  fclose(ofile);
  fclose(file);

  // Free
  free(header);
  free(hbuf);
  free(fbuf);
  free(c);
  cudaFree(dbuf);
  cudaFree(dfbuf);
  cudaFree(cp1);
  cudaFree(cp2);
  cudaFree(dc);

  // Free plan
  cufftDestroy(ftc2cf);
  cufftDestroy(ftc2cb);

  return 0;
}
