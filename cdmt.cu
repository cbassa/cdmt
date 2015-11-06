#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "hdf5.h"

#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10

struct header {
  int64_t headersize,buffersize;
  unsigned int nchan,nsamp,nbit,nif,nsub;
  int machine_id,telescope_id,nbeam,ibeam,sumif;
  double tstart,tsamp,fch1,foff,fcen,bwchan;
  double src_raj,src_dej,az_start,za_start;
  char source_name[80],ifstream[8],inpfile[80];
  char *rawfname[4];
};

struct header read_h5_header(char *fname);
void get_channel_chirp(double fcen,double bw,float dm,int nchan,int nbin,int nsub,cufftComplex *c);
__global__ void transpose_unpadd_and_detect(cufftComplex *cp1,cufftComplex *cp2,int nbin,int nchan,int nfft,int nsub,int noverlap,int nsamp,float *fbuf);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b);
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,float scale);
__global__ void unpack_and_padd(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2);
__global__ void swap_spectrum_halves(cufftComplex *cp1,cufftComplex *cp2,int nx,int ny);
void write_filterbank_header(struct header h,FILE *file);

int main(int argc,char *argv[])
{
  int i,nsamp,nfft,mbin,nvalid,nchan=8,nbin=65536,noverlap=2048,nsub=20;
  int iblock,nread;
  char *header,*h5buf[4],*dh5buf[4];
  FILE *file[4],*ofile;
  float *fbuf,*dfbuf;
  cufftComplex *cp1,*cp2,*dc,*c;
  cufftHandle ftc2cf,ftc2cb;
  int idist,odist,iembed,oembed,istride,ostride;
  dim3 blocksize,gridsize;
  struct header h5;
  clock_t startclock;

  // Read HDF5 header
  h5=read_h5_header(argv[1]);

  // Set number of subbands
  nsub=h5.nsub;

  // Adjust header for filterbank format
  h5.tsamp*=nchan;
  h5.nchan=nsub*nchan;
  h5.nbit=32;
  h5.fch1=h5.fcen+0.5*h5.nsub*h5.bwchan-0.5*h5.bwchan/nchan;
  h5.foff=-fabs(h5.bwchan/nchan);

  // Allocate chirp
  c=(cufftComplex *) malloc(sizeof(cufftComplex)*nbin*nsub);

  // Compute chirp
  get_channel_chirp(h5.fcen,nsub*h5.bwchan,39.659298,nchan,nbin,nsub,c);

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
  for (i=0;i<4;i++) {
    h5buf[i]=(char *) malloc(sizeof(char)*nsamp*nsub);
    checkCudaErrors(cudaMalloc((void **) &dh5buf[i],sizeof(char)*nsamp*nsub));
  }

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

  // Open output file
  ofile=fopen("fifo","w");

  // Write filterbank header
  write_filterbank_header(h5,ofile);

  // Read files
  for (i=0;i<4;i++) 
    file[i]=fopen(h5.rawfname[i],"r");

  // Loop over input file contents
  for (iblock=0;;iblock++) {
    // Read block
    startclock=clock();
    for (i=0;i<4;i++)
      nread=fread(h5buf[i],sizeof(char),nsamp*nsub,file[i])/nsub;
    if (nread==0)
      break;
    printf("Block: %d: Read %d MB in %.2f s\n",iblock,sizeof(char)*nread*nsub*4/(1<<20),(float) (clock()-startclock)/CLOCKS_PER_SEC);

    // Copy buffers to device
    startclock=clock();
    for (i=0;i<4;i++)
      checkCudaErrors(cudaMemcpy(dh5buf[i],h5buf[i],sizeof(char)*nread*nsub,cudaMemcpyHostToDevice));

    // Unpack data and padd data
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
    unpack_and_padd<<<gridsize,blocksize>>>(dh5buf[0],dh5buf[1],dh5buf[2],dh5buf[3],nread,nbin,nfft,nsub,noverlap,cp1,cp2);

    // Perform FFTs
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_FORWARD));

    // Swap spectrum halves for large FFTs
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft*nsub/blocksize.y+1; gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,nbin,nfft*nsub);

    // Perform complex multiplication of FFT'ed data with chirp (in place)
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin*nsub/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=1;
    PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp1,dc,cp1,nbin*nsub,nfft,1.0/(float) nbin);
    PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp2,dc,cp2,nbin*nsub,nfft,1.0/(float) nbin);

    // Swap spectrum halves for small FFTs
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan*nfft*nsub/blocksize.y+1; gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,mbin,nchan*nfft*nsub);

    // Perform FFTs
    checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_INVERSE));
    checkCudaErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_INVERSE));

    // Detect data
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=nfft/blocksize.z+1;
    transpose_unpadd_and_detect<<<gridsize,blocksize>>>(cp1,cp2,mbin,nchan,nfft,nsub,noverlap/nchan,nread/nchan,dfbuf);
    
    // Copy buffer to host
    checkCudaErrors(cudaMemcpy(fbuf,dfbuf,sizeof(float)*nread*nsub,cudaMemcpyDeviceToHost));
    printf("Executed kernels in %.2f s\n",(float) (clock()-startclock)/CLOCKS_PER_SEC);

    // Write buffer
    startclock=clock();
    fwrite(fbuf,sizeof(float),nread*nsub,ofile);
    printf("Write %d MB in %.2f s\n",nread*nsub*sizeof(float)/(1<<20),(float) (clock()-startclock)/CLOCKS_PER_SEC);
  }

  // Close files
  fclose(ofile);
  for (i=0;i<4;i++)
    fclose(file[i]);

  // Free
  free(header);
  for (i=0;i<4;i++) {
    free(h5buf[i]);
    cudaFree(dh5buf);
    free(h5.rawfname[i]);
  }
  free(fbuf);
  free(c);
  cudaFree(dfbuf);
  cudaFree(cp1);
  cudaFree(cp2);
  cudaFree(dc);

  // Free plan
  cufftDestroy(ftc2cf);
  cufftDestroy(ftc2cb);

  return 0;
}

// This is a simple H5 reader for complex voltage data. Very little
// error checking is done.
struct header read_h5_header(char *fname)
{
  int i,len;
  struct header h;
  hid_t file_id,attr_id,sap_id,beam_id,memtype,group_id,space,coord_id;
  char *string,*pch;
  const char *stokes[]={"_S0_","_S1_","_S2_","_S3_"};
  char *froot,*fpart,*ftest;
  FILE *file;

  // Find filenames
  for (i=0;i<4;i++) {
    pch=strstr(fname,stokes[i]);
    if (pch!=NULL)
      break;
  }
  len=strlen(fname)-strlen(pch);
  froot=(char *) malloc(sizeof(char)*len);
  fpart=(char *) malloc(sizeof(char)*(strlen(pch)-7));
  ftest=(char *) malloc(sizeof(char)*(len+10));
  strncpy(froot,fname,len);
  strncpy(fpart,pch+4,strlen(pch)-7);

  // Check files
  for (i=0;i<4;i++) {
    // Format file name
    sprintf(ftest,"%s_S%d_%s.raw",froot,i,fpart);

    // Try to open
    if ((file=fopen(ftest,"r"))!=NULL) {
      fclose(file);
    } else {
      fprintf(stderr,"Raw file %s not found\n",ftest);
      exit (-1);
    }
    h.rawfname[i]=(char *) malloc(sizeof(char)*strlen(ftest));
    strcpy(h.rawfname[i],ftest);
  }

  // Free
  free(froot);
  free(fpart);
  free(ftest);

  // Open file
  file_id=H5Fopen(fname,H5F_ACC_RDONLY,H5P_DEFAULT);

  // Open subarray pointing group
  sap_id=H5Gopen(file_id,"SUB_ARRAY_POINTING_000",H5P_DEFAULT);

  // Start MJD
  attr_id=H5Aopen(sap_id,"EXPTIME_START_MJD",H5P_DEFAULT);
  H5Aread(attr_id,H5T_IEEE_F64LE,&h.tstart);
  H5Aclose(attr_id);

  // Declination
  attr_id=H5Aopen(sap_id,"POINT_DEC",H5P_DEFAULT);
  H5Aread(attr_id,H5T_IEEE_F64LE,&h.src_dej);
  H5Aclose(attr_id);

  // Right ascension
  attr_id=H5Aopen(sap_id,"POINT_RA",H5P_DEFAULT);
  H5Aread(attr_id,H5T_IEEE_F64LE,&h.src_raj);
  H5Aclose(attr_id);

  // Open beam 0
  beam_id=H5Gopen(sap_id,"BEAM_000",H5P_DEFAULT);

  // Number of samples
  attr_id=H5Aopen(beam_id,"NOF_SAMPLES",H5P_DEFAULT);
  H5Aread(attr_id,H5T_STD_U32LE,&h.nsamp);
  H5Aclose(attr_id);

  // Center frequency
  attr_id=H5Aopen(beam_id,"BEAM_FREQUENCY_CENTER",H5P_DEFAULT);
  H5Aread(attr_id,H5T_IEEE_F64LE,&h.fcen);
  H5Aclose(attr_id);

  // Center frequency unit
  attr_id=H5Aopen(beam_id,"BEAM_FREQUENCY_CENTER_UNIT",H5P_DEFAULT);
  memtype=H5Tcopy(H5T_C_S1);
  H5Tset_size(memtype,H5T_VARIABLE);
  H5Aread(attr_id,memtype,&string);
  H5Aclose(attr_id);
  if (strcmp(string,"Hz")==0)
    h.fcen/=1e6;

  // Channel bandwidth
  attr_id=H5Aopen(beam_id,"CHANNEL_WIDTH",H5P_DEFAULT);
  H5Aread(attr_id,H5T_IEEE_F64LE,&h.bwchan);
  H5Aclose(attr_id);

  // Center frequency unit
  attr_id=H5Aopen(beam_id,"CHANNEL_WIDTH_UNIT",H5P_DEFAULT);
  memtype=H5Tcopy(H5T_C_S1);
  H5Tset_size(memtype,H5T_VARIABLE);
  H5Aread(attr_id,memtype,&string);
  H5Aclose(attr_id);
  if (strcmp(string,"Hz")==0)
    h.bwchan/=1e6;

  // Get source
  attr_id=H5Aopen(beam_id,"TARGETS",H5P_DEFAULT);
  memtype=H5Tcopy(H5T_C_S1);
  H5Tset_size(memtype,H5T_VARIABLE);
  H5Aread(attr_id,memtype,&string);
  H5Aclose(attr_id);
  strcpy(h.source_name,string);

  // Open coordinates
  coord_id=H5Gopen(beam_id,"COORDINATES",H5P_DEFAULT);

  // Open coordinate 0
  group_id=H5Gopen(coord_id,"COORDINATE_0",H5P_DEFAULT);

  // Sampling time
  attr_id=H5Aopen(group_id,"INCREMENT",H5P_DEFAULT);
  H5Aread(attr_id,H5T_IEEE_F64LE,&h.tsamp);
  H5Aclose(attr_id);

  // Close group
  H5Gclose(group_id);

  // Open coordinate 1
  group_id=H5Gopen(coord_id,"COORDINATE_1",H5P_DEFAULT);

  // Number of subbands
  attr_id=H5Aopen(group_id,"AXIS_VALUES_WORLD",H5P_DEFAULT);
  space=H5Aget_space(attr_id);
  h.nsub=H5Sget_simple_extent_npoints(space);
  H5Aclose(attr_id);

  // Close group
  H5Gclose(group_id);

  // Close coordinates
  H5Gclose(coord_id);

  // Close beam, sap and file
  H5Gclose(beam_id);
  H5Gclose(sap_id);
  H5Fclose(file_id);

  return h;
}

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
__global__ void unpack_and_padd(char *dbuf0,char *dbuf1,char *dbuf2,char *dbuf3,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
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
      cp1[idx1].x=(float) dbuf0[idx2];
      cp1[idx1].y=(float) dbuf1[idx2];
      cp2[idx1].x=(float) dbuf2[idx2];
      cp2[idx1].y=(float) dbuf3[idx2];
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

void send_string(char *string,FILE *file)
{
  int len;

  len=strlen(string);
  fwrite(&len,sizeof(int),1,file);
  fwrite(string,sizeof(char),len,file);

  return;
}

void send_float(char *string,float x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(float),1,file);

  return;
}

void send_int(char *string,int x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(int),1,file);

  return;
}

void send_double(char *string,double x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(double),1,file);

  return;
}

double dec2sex(double x)
{
  double d,sec,min,deg;
  char sign;
  char tmp[32];

  sign=(x<0 ? '-' : ' ');
  x=3600.0*fabs(x);

  sec=fmod(x,60.0);
  x=(x-sec)/60.0;
  min=fmod(x,60.0);
  x=(x-min)/60.0;
  deg=x;

  sprintf(tmp,"%c%02d%02d%09.6lf",sign,(int) deg,(int) min,sec);
  sscanf(tmp,"%lf",&d);

  return d;
}

void write_filterbank_header(struct header h,FILE *file)
{
  double ra,de;


  ra=dec2sex(h.src_raj/15.0);
  de=dec2sex(h.src_dej);
  
  send_string("HEADER_START",file);
  send_string("rawdatafile",file);
  send_string(h.rawfname[0],file);
  send_string("source_name",file);
  send_string(h.source_name,file);
  send_int("machine_id",11,file);
  send_int("telescope_id",11,file);
  send_double("src_raj",ra,file);
  send_double("src_dej",de,file);
  send_int("data_type",1,file);
  send_double("fch1",h.fch1,file);
  send_double("foff",h.foff,file);
  send_int("nchans",160,file);
  send_int("nbeams",0,file);
  send_int("ibeam",0,file);
  send_int("nbits",h.nbit,file);
  send_double("tstart",h.tstart,file);
  send_double("tsamp",h.tsamp,file);
  send_int("nifs",1,file);
  send_string("HEADER_END",file);

  return;
}
