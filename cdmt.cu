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
void get_channel_chirp(double fcen,double bw,float dm,int nchan,int nbin,int nsub,cufftComplex *c)
{
  int ibin,ichan,isub,mbin,idx;
  double s,rt,t,f,fsub,fchan,bwchan,bwsub;

  // Main constant
  s=2.0*M_PI*dm/DMCONSTANT;

  // Number of channels per subband
  mbin=nbin/nchan;

  // Subband bandwidth
  bwsub=bw/nsub;

  // Channel bandwidth
  bwchan=bw/(nchan*nsub);

  // Loop over subbands
  for (isub=0;isub<nsub;isub++) {
    // Subband frequency
    fsub=fcen-0.5*bw+bw*(float) isub/(float) nsub+0.5*bw/(float) nsub;

    // Loop over channels
    for (ichan=0;ichan<nchan;ichan++) {
      // Channel frequency
      fchan=fsub-0.5*bwsub+bwsub*(float) ichan/(float) nchan+0.5*bwsub/(float) nchan;
      
      // Loop over bins in channel
      for (ibin=0;ibin<mbin;ibin++) {
	// Bin frequency
	f=-0.5*bwchan+bwchan*(float) ibin/(float) mbin+0.5*bwchan/(float) mbin;
	
	// Phase delay
	rt=-f*f*s/((fchan+f)*fchan*fchan);

	// Taper
	t=1.0/sqrt(1.0+pow((f/(0.47*bwchan)),80));
	
	// Index
	idx=ibin+ichan*mbin+isub*mbin*nchan;

	// Chirp
	c[idx].x=cos(rt)*t;
	c[idx].y=sin(rt)*t;
      }
    }
  }

  return;
}

// Scale cufftComplex 
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s)
{
  cufftComplex c;
  c.x=s*a.x;
  c.y=s*a.y;
  return c;
}

// Complex multiplication
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

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are padded with noverlap samples on either side for the
// convolution.
__global__ void unpack_and_padd(char *dbuf,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    idx1=ibin+nbin*isub+nsub*nbin*ifft;
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    idx2=isub+nsub*isamp;
    if (isamp<0 || isamp>=nsamp) {
      cp1[idx1].x=0.0;
      cp1[idx1].y=0.0;
      cp2[idx1].x=0.0;
      cp2[idx1].y=0.0;
    } else {
      cp1[idx1].x=(float) dbuf[4*idx2];
      cp1[idx1].y=(float) dbuf[4*idx2+1];
      cp2[idx1].x=(float) dbuf[4*idx2+2];
      cp2[idx1].y=(float) dbuf[4*idx2+3];
    }
  }

  return;
}

// Since complex-to-complex FFTs put the center frequency at bin zero
// in the frequency domain, the two halves of the spectrum need to be
// swapped.
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

// After the segmented FFT the data is in a cube of nbin by nchan by
// nfft, where nbin and nfft are the time indices. Here we rearrange
// the 3D data cube into a 2D array of frequency and time, while also
// removing the overlap regions and detecting (generating Stokes I).
__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nbin,int nchan,int nfft,int nsub,int noverlap,int nsamp,float *fbuf)
{
  int64_t ibin,ichan,ifft,isub,isamp,idx1,idx2;
  
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ichan=blockIdx.y*blockDim.y+threadIdx.y;
  ifft=blockIdx.z*blockDim.z+threadIdx.z;
  if (ibin<nbin && ichan<nchan && ifft<nfft) {
    // Loop over subbands
    for (isub=0;isub<nsub;isub++) {
      // Padded array index
      //      idx1=ibin+nbin*isub+nsub*nbin*(ichan+nchan*ifft);
      idx1=ibin+ichan*nbin+(nsub-isub-1)*nbin*nchan+ifft*nbin*nchan*nsub;

      // Time index
      isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
      
      // Output array index
      idx2=(nchan-ichan-1)+isub*nchan+nsub*nchan*isamp;
      
      // Select data points from valid region
      if (ibin>=noverlap && ibin<=nbin-noverlap && isamp>=0 && isamp<nsamp)
	fbuf[idx2]=cp1[idx1].x*cp1[idx1].x+cp1[idx1].y*cp1[idx1].y+cp2[idx1].x*cp2[idx1].x+cp2[idx1].y*cp2[idx1].y;
    }
  }

  return;
}

int main(int argc,char *argv[])
{
  int nsamp,nfft,mbin,nvalid,nchan=8,nbin=65536,noverlap=2048,nsub=20;
  int iblock,nread;
  char *header,*hbuf,*dbuf;
  FILE *file,*ofile;
  float *fbuf,*dfbuf;
  cufftComplex *cp1,*cp2,*dc,*c;
  cufftHandle ftc2cf,ftc2cb;
  int idist,odist,iembed,oembed,istride,ostride;
  dim3 blocksize,gridsize;

  c=(cufftComplex *) malloc(sizeof(cufftComplex)*nbin*nsub);

  // Compute chirp
  get_channel_chirp(119.62890625,nsub*0.1953125,39.659298,nchan,nbin,nsub,c);

  // Data size
  nvalid=nbin-2*noverlap;
  nsamp=200*nvalid;
  nfft=(int) ceil(nsamp/(float) nvalid);
  mbin=nbin/nchan;

  printf("nbin: %d nfft: %d nsub: %d mbin: %d nchan: %d nsamp: %d nvalid: %d\n",nbin,nfft,nsub,mbin,nchan,nsamp,nvalid);

  // Allocate memory for complex timeseries
  checkCudaErrors(cudaMalloc((void **) &cp1,sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp2,sizeof(cufftComplex)*nbin*nfft*nsub));

  // Allocate device memory for chirp
  checkCudaErrors(cudaMalloc((void **) &dc,sizeof(cufftComplex)*nbin*nsub));

  // Allocate memory for redigitized output and header
  header=(char *) malloc(sizeof(char)*HEADERSIZE);
  hbuf=(char *) malloc(sizeof(char)*4*nsamp*nsub);
  checkCudaErrors(cudaMalloc((void **) &dbuf,sizeof(char)*4*nsamp*nsub));

  // Allocate output buffers
  fbuf=(float *) malloc(sizeof(float)*nsamp*nsub);
  checkCudaErrors(cudaMalloc((void **) &dfbuf,sizeof(float)*nsamp*nsub));

  // Generate FFT plan (batch in-place forward FFT)
  idist=nbin;  odist=nbin;  iembed=nbin;  oembed=nbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cf,1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nfft*nsub));

  // Generate FFT plan (batch in-place backward FFT)
  idist=mbin;  odist=mbin;  iembed=mbin;  oembed=mbin;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2cb,1,&mbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nchan*nfft*nsub));

  // Copy chirp to device
  checkCudaErrors(cudaMemcpy(dc,c,sizeof(cufftComplex)*nbin*nsub,cudaMemcpyHostToDevice));

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
    nread=fread(hbuf,sizeof(char),4*nsamp*nsub,file)/(4*nsub);
    if (nread==0)
      break;

    // Copy buffer to device
    checkCudaErrors(cudaMemcpy(dbuf,hbuf,sizeof(char)*4*nread*nsub,cudaMemcpyHostToDevice));

    // Unpack data and padd data
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1;
    gridsize.y=nfft/blocksize.y+1;
    gridsize.z=nsub/blocksize.z+1;
    unpack_and_padd<<<gridsize,blocksize>>>(dbuf,nread,nbin,nfft,nsub,noverlap,cp1,cp2);

    // Perform FFTs
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_FORWARD));

    // Swap spectrum halves for large FFTs
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1;
    gridsize.y=nfft*nsub/blocksize.y+1;
    gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,nbin,nfft*nsub);

    // Perform complex multiplication of FFT'ed data with chirp (in place)
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=nbin*nsub/blocksize.x+1;
    gridsize.y=nfft/blocksize.y+1;
    gridsize.z=1;
    PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp1,dc,cp1,nbin*nsub,nfft,1.0/(float) nbin);
    PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp2,dc,cp2,nbin*nsub,nfft,1.0/(float) nbin);

    // Swap spectrum halves for small FFTs
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=mbin/blocksize.x+1;
    gridsize.y=nchan*nfft*nsub/blocksize.y+1;
    gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,mbin,nchan*nfft*nsub);

    // Perform FFTs
    checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_INVERSE));
    checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_INVERSE));

    // Detect data
    blocksize.x=32;
    blocksize.y=32;
    blocksize.z=1;
    gridsize.x=mbin/blocksize.x+1;
    gridsize.y=nchan/blocksize.y+1;
    gridsize.z=nfft/blocksize.z+1;
    transpose_unpadd_and_detect<<<gridsize,blocksize>>>(cp1,cp2,mbin,nchan,nfft,nsub,noverlap/nchan,nread/nchan,dfbuf);
    
    // Copy buffer to host
    checkCudaErrors(cudaMemcpy(fbuf,dfbuf,sizeof(float)*nread*nsub,cudaMemcpyDeviceToHost));

    // Write buffer
    fwrite(fbuf,sizeof(float),nread*nsub,ofile);
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
