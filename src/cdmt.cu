#include "cdmt.h"

int main(int argc,char *argv[])
{
  int arg=0;             // Command line inputs
  int i,idm,iblock;      // Iterators
  int rv;                // Return value
  clock_t startclock;    // Clock time
  size_t cfsize, cbsize; // FFT sizes
  size_t minfftsize;     // Min of cfsize and cbsize
  size_t gpu_mems[2];    // Used/total GPU VRAM

  // VDIF file input
  struct vdif_file *vf;

  // File input
  char *hdrfname;        // Name of first HDF5 metadata file
  FILE *file;            // Pointer to first HDF5 metadata file
  FILE **input_files;    // Pointers to array of raw file names
  char fheader[1024];    // Filterbank header string
  int nread,nread_tmp;   // Amount of data read from block in bytes
  int bytes_read;        // Filterbank header size in bytes
  char *vfbuf;           // Host buffer for reading VDIF data
  char *tmp_vfbuf;       // Host buffer for reading VDIF data
  char *dvfbuf;          // Device buffer for reading VDIF data
  
  // File output
  char fname[256];         // Name of output filterbanks
  char obsid[128]="cdmt";  // Prefix of the output filenames
  char src_name[256]="\0"; // Source name override
  FILE **output_files;     // Pointer to filterbank output_filess
  float *dfbuf;            // Device buffer for output
  unsigned char *dcbuf;    // Device buffer for redigitised output
  unsigned char *cbuf;     // Host buffer for redigitised output

  // DMs
  float *dm,*ddm;      // Host/device arrays for DM steps
  float dm_start=-1;   // Start DM
  float dm_step=-1;    // DM step size
  int ndm=-1;          // Number of DM steps

  // Forward FFT
  int nforward=128;   // Number of forward FFTs per cuFFT call
  int nsub=24;        // Number of subbands
  int nchan=32;       // Number of channels per subband
  int nbin=32768;     // Size of forward FFT
  int noverlap=1024;  // Size of the overlap region
  int nvalid;         // Number of non-overlapping bins
  int nsamp;          // Number of samples per block
  int nframe;         // Number of VDIF frames per block
  int nfft;           // Number of parallel FFTs
  int ndec=1;         // Number of time samples to average

  // Backward FFT
  int mchan;          // Number of filterbank channels (nsub*nchan)
  int mbin;           // Size of backward FFT (nbin/nchan)
  int msamp;          // Number of block samples per channel (nsamp/nchan)
  int msum=1024;      // Size of block sum
  int mblock;         // Number of blocks (msamp/msum)

  // CUDA
  int device=0;             // GPU device code
  cufftComplex *cp1p,*cp2p; // Complex timeseries
  cufftComplex *cp1,*cp2;   // Dedispersed complex timeseries
  cufftComplex *dc;         // Chirp
  float *bs1,*bs2;          // Block sums
  float *zavg,*zstd;        // Channel averages and standard deviations
  cufftHandle ftc2cf;       // Forward FFT plan
  cufftHandle ftc2cb;       // Backward FFT plan
  int idist,odist,iembed,oembed,istride,ostride;  // FFT plan params
  dim3 blocksize,gridsize;  // GPU mapping params

  // Read options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"hd:D:b:N:n:f:s:c:m:o:p:"))!=-1) {
      switch (arg) {

        case 'h':
          usage();
          return 0;

        case 'd':
          sscanf(optarg,"%f,%f,%d",&dm_start,&dm_step,&ndm);
          break;

        case 'D':
          device=atoi(optarg);
          break;
        
        case 'b':
          ndec=atoi(optarg);
          break;

        case 'N':
          nbin=atoi(optarg);
          break;
	
        case 'n':
          noverlap=atoi(optarg);
          break;

        case 'f':
          nforward=atoi(optarg);
          break;

        case 's':
          nsub=atoi(optarg);
          break;

        case 'c':
          nchan=atoi(optarg);
          break;

        case 'm':
          msum=atoi(optarg);
          break;

        case 'o':
          strcpy(obsid,optarg);
          break;

        case 'p':
          strcpy(src_name,optarg);
          break;

        default:
          return 1;
      }
    }
  } else {
    usage();
    return 0;
  }

  // Check required inputs were given
  if ((dm_start==-1)||(dm_step==-1)||(ndm==-1)) {
    fprintf(stderr, "ERROR :: DM parameters were not specified. Exiting.\n");
    return 1;
  }
  if (argc<=optind) {
    fprintf(stderr, "ERROR :: Failed to provide an input file. Exiting.\n");
    return 1;
  }
  hdrfname=argv[optind];

  // Basic input checks
  if (dm_start<0.0) {
    fprintf(stderr, "ERROR :: Start DM must be a non-negative number (currently %f). Exiting.\n", dm_start);
    return 1;
  }
  if (dm_step<=0.0) {
    fprintf(stderr, "ERROR :: DM step size must be a positive number (currently %f). Exiting.\n", dm_step);
    return 1;
  }
  if (ndm<1) {
    fprintf(stderr, "ERROR :: Number of DM trials must be a positive integer (currently %d). Exiting.\n", ndm);
    return 1;
  }
  if (ndec<1) {
    fprintf(stderr, "ERROR :: Number of averages time samples must be a positive integer (currently %d). Exiting.\n", ndec);
    return 1;
  }
  if (nbin<1) {
    fprintf(stderr, "ERROR :: FFT size must be a positive integer (currently %d). Exiting.\n", nbin);
    return 1;
  }
  if (noverlap<1) {
    fprintf(stderr, "ERROR :: FFT overlap must be a positive integer (currently %d). Exiting.\n", noverlap);
    return 1;
  }
  if (nforward<1) {
    fprintf(stderr, "ERROR :: Number of FFTs must be a positive integer (currently %d). Exiting.\n", nforward);
    return 1;
  }
  if (nsub<1) {
    fprintf(stderr, "ERROR :: Number of subbands must be a positive integer (currently %d). Exiting.\n", nsub);
    return 1;
  }
  if (nchan<1) {
    fprintf(stderr, "ERROR :: Channelisation factor must be a positive integer (currently %d). Exiting.\n", nchan);
    return 1;
  }
  if (msum<1) {
    fprintf(stderr, "ERROR :: Size of blocksum must be a positive integer (currently %d). Exiting.\n", msum);
    return 1;
  }

  // Sanity checks
  if (nbin % nchan != 0) {
    fprintf(stderr, "ERROR :: nbin must be divisible by nchan (%d) (currently %d, remainder: %d). Exiting.\n", nchan, nbin, nbin % nchan);
    return 1;
  }
  if (nbin-2*noverlap < 1) {
    fprintf(stderr, "ERROR :: FFT size (%d) must be greater than twice the FFT overlap (%d). Exiting.\n", nbin, noverlap);
    return 1;
  }
  if ((nforward * (nbin-2*noverlap)) % nchan != 0) {
    fprintf(stderr, "ERROR :: Number of valid samples must be divisible by nchan (%d) (currently %d, remainer %d). Exiting.\n", nchan, nbin-2*noverlap, (nbin-2*noverlap) % nchan);
    return 1;
  }
  if ((nforward * (nbin-2*noverlap) / nchan) % msum != 0) {
    fprintf(stderr, "ERROR :: Number of valid samples must be divisible by msum (%d) (currently %d, remainder %d).\n", msum, (nforward * (nbin-2*noverlap) / nchan), (nforward * (nbin-2*noverlap) / nchan) % msum);
    return 1;
  }
  if ((nforward * (nbin-2*noverlap)) % 128 != 0) {
    fprintf(stderr, "ERROR :: Number of valid samples must be divisible by the number of time samples per frame (128) (currently %d, remainder %d). Exiting.\n", (nforward * (nbin-2*noverlap)), (nforward * (nbin-2*noverlap)) % 128);
    return 1;
  }

  // File checks
   if (access(hdrfname, F_OK)==-1)
  {
    fprintf(stderr, "ERROR :: Input file does not exist (%s). Exiting.\n", hdrfname);
    return 1;
  }
  if (access(hdrfname, R_OK)==-1)
  {
    fprintf(stderr, "ERROR :: Input file is not readable (%s). Exiting.\n", hdrfname);
    return 1;
  }

  // Initialise struct of VDIF file information
  vf=init_vdif_struct(nsub);

  // Read data from ascii header and store it in VDIF struct
  rv=parse_ascii_header(vf,nsub,hdrfname,src_name);
  if (rv!=0) {
    free_vdif_struct(vf,nsub);
    return 1;
  }

  // Update header information
  vf->nsub=nsub;
  vf->tsamp*=nchan*ndec;
  vf->nchan=nsub*nchan;
  vf->nbit=8;
  vf->fch1=vf->fcen+0.5*vf->nsub*vf->bwchan-0.5*vf->bwchan/nchan;
  vf->foff=-fabs(vf->bwchan/nchan);

  // Check that the FFT size and overlap size are large enough
  const double  stg1 = (1.0 / 2.41e-4) * abs(pow((double) vf->fch1 + nsub * vf->foff + vf->foff *0.5,-2.0) - pow((double) vf->fch1 + vf->nsub * vf->foff - vf->foff *0.5, -2.0)) * (dm_start + dm_step * (ndm - 1));
  const int overlap_check = (int) (stg1 / vf->tsamp);
  if (overlap_check > nbin) {
    fprintf(stderr, "WARNING :: The size of your FFT bin is too short for the given DMs and frequencies. Given bin size: %d, Suggested minimum bin size: %d (maximum dispersion delay %f).\n", nbin, overlap_check, stg1);
  } else if (overlap_check / 2 > noverlap) {
    fprintf(stderr, "WARNING :: The size of your FFT overlap is too short for the given maximum DM. Given overlap: %d, Suggested minimum overlap: %d (maximum dispersion delay %f).\n", noverlap, overlap_check / 2, stg1);
  }

  // Open input data files
  input_files=(FILE **) malloc(sizeof(FILE *)*nsub);
  for (i=0;i<nsub;i++) {
    printf("Opening file %s\n", vf->datafns[i]);
    input_files[i]=fopen(vf->datafns[i],"r");
    if (input_files[i]==NULL) {
      fprintf(stderr, "ERROR :: Input file failed to open (null pointer). Exiting.\n");
      return 1;
    }
  }

  // Data sizes
  nvalid=nbin-2*noverlap;
  nsamp=nforward*nvalid;
  nframe=4*nsamp/VDIF_DATA_BYTES;
  nfft=(int) ceil(nsamp/(float) nvalid);
  mbin=nbin/nchan;  // nbin must be divisible by nchan
  mchan=nsub*nchan;
  msamp=nsamp/nchan;  // nforward * nvalid must be divisible by nchan
  mblock=msamp/msum;  // nforward * nvalid / nchan must be divisible by msum

  printf("\nUsing the following parameters:\n\n");
  printf("         Num of subbands = %d\n", nsub);
  printf("   Channelisation factor = %d\n", nchan);
  printf("     Downsampling factor = %d\n", ndec);
  printf("        Forward FFT size = %d\n", nbin);
  printf("       Backward FFT size = %d\n", mbin);
  printf("   Valid samples per FFT = %d\n", nvalid);
  printf("             FFTs per op = %d\n", nfft);
  printf("  Samples per forward op = %d\n", nsamp);
  printf(" Samples per backward op = %d\n", msamp);
  printf("       Size of block sum = %d\n", msum);
  printf("       Num of block sums = %d\n\n", mblock);

  // Set device
  checkCudaErrors(cudaSetDevice(device));

  // Generate FFT plan (batch in-place forward FFT)
  idist=nbin;  odist=nbin;  iembed=nbin;  oembed=nbin;  istride=1;  ostride=1;
  checkCufftErrors(cufftPlanMany(&ftc2cf,1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nfft*nsub));
  checkCufftErrors(cufftGetSizeMany(ftc2cf,1,&nbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nfft*nsub,&cfsize));

  // Generate FFT plan (batch in-place backward FFT)
  idist=mbin;  odist=mbin;  iembed=mbin;  oembed=mbin;  istride=1;  ostride=1;
  checkCufftErrors(cufftPlanMany(&ftc2cb,1,&mbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nchan*nfft*nsub));
  checkCufftErrors(cufftGetSizeMany(ftc2cb,1,&mbin,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2C,nchan*nfft*nsub,&cbsize));

  // Get the maximum size needed for the FFT operations (they should be the same, check for safety)
  minfftsize = cfsize > cbsize ? cfsize : cbsize;
  printf("Allocated %ld MB for cuFFT work (saving %ld MB)\n", minfftsize >> 20, (cfsize + cbsize - minfftsize) >> 20);

  // Predict the overall VRAM usage
  long unsigned int bytes_used=\
      sizeof(char)*nsamp*nsub*4 \
    + sizeof(cufftComplex)*nbin*nfft*nsub*4 \
    + sizeof(cufftComplex)*nbin*nsub*ndm \
    + sizeof(float)*nsamp*nsub \
    + sizeof(float)*mblock*mchan*2 \
    + sizeof(float)*mchan*2 \
    + sizeof(unsigned char)*msamp*mchan/ndec \
    + sizeof(float)*ndm;

  // Get the total / available VRAM
  checkCudaErrors(cudaMemGetInfo(&(gpu_mems[0]), &(gpu_mems[1])));
  printf("Preparing for GPU memory allocations. Current memory usage: %ld / %ld GB\n", (gpu_mems[1] - gpu_mems[0]) >> 30, gpu_mems[1] >> 30);
  printf("We anticipate %ld MB (%ld GB) to be allocated on the GPU (%ld MB for cuFFT planning).\n", (bytes_used + minfftsize) >> 20, (bytes_used + minfftsize) >> 30, minfftsize >> 20);

  // Allocate memory for raw complex timeseries
  vfbuf=(char *) malloc(sizeof(char)*nsamp*nsub*4);
  tmp_vfbuf=(char *) malloc(nframe*VDIF_FRAME_BYTES);
  checkCudaErrors(cudaMalloc((void **) &dvfbuf,sizeof(char)*nsamp*nsub*4));

  // Allocate memory for unpacked complex timeseries
  checkCudaErrors(cudaMalloc((void **) &cp1,sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp2,sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp1p,sizeof(cufftComplex)*nbin*nfft*nsub));
  checkCudaErrors(cudaMalloc((void **) &cp2p,sizeof(cufftComplex)*nbin*nfft*nsub));

  // Allocate device memory for chirp
  checkCudaErrors(cudaMalloc((void **) &dc,sizeof(cufftComplex)*nbin*nsub*ndm));

  // Allocate memory for detected data
  checkCudaErrors(cudaMalloc((void **) &dfbuf,sizeof(float)*nsamp*nsub));

  // Allocate device memory for block sums
  checkCudaErrors(cudaMalloc((void **) &bs1,sizeof(float)*mblock*mchan));
  checkCudaErrors(cudaMalloc((void **) &bs2,sizeof(float)*mblock*mchan));

  // Allocate device memory for channel averages and standard deviations
  checkCudaErrors(cudaMalloc((void **) &zavg,sizeof(float)*mchan));
  checkCudaErrors(cudaMalloc((void **) &zstd,sizeof(float)*mchan));

  // Allocate output buffers for redigitized data
  cbuf=(unsigned char *) malloc(sizeof(unsigned char)*msamp*mchan/ndec);
  checkCudaErrors(cudaMalloc((void **) &dcbuf,sizeof(unsigned char)*msamp*mchan/ndec));

  // Allocate DMs and copy to device
  dm=(float *) malloc(sizeof(float)*ndm);
  for (idm=0;idm<ndm;idm++)
    dm[idm]=dm_start+(float) idm*dm_step;
  checkCudaErrors(cudaMalloc((void **) &ddm,sizeof(float)*ndm));
  checkCudaErrors(cudaMemcpy(ddm,dm,sizeof(float)*ndm,cudaMemcpyHostToDevice));

  // Allow memory alloation/copy actions to finish before processing
  cudaDeviceSynchronize();

  // Compute chirp
  blocksize.x=32; blocksize.y=32; blocksize.z=1;
  gridsize.x=nsub/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=ndm/blocksize.z+1;
  compute_chirp<<<gridsize,blocksize>>>(vf->fcen,nsub*vf->bwchan,ddm,nchan,nbin,nsub,ndm,dc);

  // Write temporary filterbank header
  file=fopen("/tmp/header.fil","w");
  if (file==NULL) {
    fprintf(stderr, "ERROR :: Unable to open /tmp/header.fil to write temporary header. Exiting.\n");
    return 1;
  }
  write_filterbank_header(vf,file);
  fclose(file);
  file=fopen("/tmp/header.fil","r");
  if (file==NULL) {
    fprintf(stderr, "ERROR :: Unable to open /tmp/header.fil to read temporary header length. Exiting.\n");
    return 1;
  }
  bytes_read=fread(fheader,sizeof(char),1024,file);
  fclose(file);
  
  // Format file names and open
  output_files=(FILE **) malloc(sizeof(FILE *)*ndm);
  for (idm=0;idm<ndm;idm++) {
    sprintf(fname,"%s_cdm%07.3f.fil",obsid,dm[idm]);

    output_files[idm]=fopen(fname,"w");
    if (output_files[idm]==NULL) {
      fprintf(stderr, "ERROR :: Unable to open output file %s. Exiting.\n", fname);
      return 1;
    }
  }

  // Write headers
  for (idm=0;idm<ndm;idm++) {
    // Send header
    fwrite(fheader,sizeof(char),bytes_read,output_files[idm]);
  }

  // Loop over input file contents
  nread=INT_MAX;
  nread_tmp=-1;
  for (iblock=0;;iblock++) {
    // Reset vfbuf memory
    memset(vfbuf,0,sizeof(char)*nsamp*nsub*4);

    // Read block
    startclock=clock();
    for (i=0;i<nsub;i++)
      nread_tmp=read_block_and_strip(input_files[i],vfbuf,tmp_vfbuf,i,nframe);
    
    // Check for error
    if (nread_tmp==-1)
      break;
    
    if (nread>nread_tmp)
      nread=nread_tmp;

    printf("Block: %d: Read %lu MB in %.2f s\n",iblock,sizeof(char)*nread*nsub*4/(1<<20),(float) (clock()-startclock)/CLOCKS_PER_SEC);

    if (nread==0) {
      printf("No data read from last block; assuming EOF, finishing up.\n");
      break;
    } else if (iblock!=0 && nread<nread_tmp) {
      printf("Received less data than expected; we may have parsed out of order data or we are nearing the EOF.\n");
    }      

    // Copy buffers to device
    startclock=clock();
    checkCudaErrors(cudaMemcpy(dvfbuf,vfbuf,sizeof(char)*nread*nsub*4,cudaMemcpyHostToDevice));

    // Unpack data and padd data
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
    if (iblock>0) {
      unpack_and_padd<<<gridsize,blocksize>>>(dvfbuf,nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
    } else {
      unpack_and_padd_first_iteration<<<gridsize,blocksize>>>(dvfbuf,nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
    }

    // Perform FFTs
    checkCufftErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp1p,(cufftComplex *) cp1p,CUFFT_FORWARD));
    checkCufftErrors(cufftExecC2C(ftc2cf,(cufftComplex *) cp2p,(cufftComplex *) cp2p,CUFFT_FORWARD));

    // Swap spectrum halves for large FFTs
    blocksize.x=32; blocksize.y=32; blocksize.z=1;
    gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft*nsub/blocksize.y+1; gridsize.z=1;
    swap_spectrum_halves<<<gridsize,blocksize>>>(cp1p,cp2p,nbin,nfft*nsub);

    // Loop over dms
    for (idm=0;idm<ndm;idm++) {

      // Perform complex multiplication of FFT'ed data with chirp
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=nbin*nsub/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=1;
      // (Removed scaling by 1/nbin to fix digitisation bug)
      PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp1p,dc,cp1,nbin*nsub,nfft,idm,1.0);
      PointwiseComplexMultiply<<<gridsize,blocksize>>>(cp2p,dc,cp2,nbin*nsub,nfft,idm,1.0);

      // Padd the next iteration
      if (idm==ndm-1) {
        blocksize.x=32; blocksize.y=32; blocksize.z=1;
        gridsize.x=nbin/blocksize.x+1; gridsize.y=nfft/blocksize.y+1; gridsize.z=nsub/blocksize.z+1;
        padd_next_iteration<<<gridsize,blocksize>>>(dvfbuf,nread,nbin,nfft,nsub,noverlap,cp1p,cp2p);
      }
      
      // Swap spectrum halves for small FFTs
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan*nfft*nsub/blocksize.y+1; gridsize.z=1;
      swap_spectrum_halves<<<gridsize,blocksize>>>(cp1,cp2,mbin,nchan*nfft*nsub);
      
      // Perform FFTs
      checkCufftErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp1,(cufftComplex *) cp1,CUFFT_INVERSE));
      checkCufftErrors(cufftExecC2C(ftc2cb,(cufftComplex *) cp2,(cufftComplex *) cp2,CUFFT_INVERSE));
      
      // Detect data
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mbin/blocksize.x+1; gridsize.y=nchan/blocksize.y+1; gridsize.z=nfft/blocksize.z+1;
      transpose_unpadd_and_detect<<<gridsize,blocksize>>>(cp1,cp2,mbin,nchan,nfft,nsub,noverlap/nchan,nread/nchan,dfbuf);
      
      // Compute block sums for redigitization
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mchan/blocksize.x+1; gridsize.y=mblock/blocksize.y+1; gridsize.z=1;
      compute_block_sums<<<gridsize,blocksize>>>(dfbuf,mchan,mblock,msum,bs1,bs2);

      // Compute channel stats
      blocksize.x=32; blocksize.y=1; blocksize.z=1;
      gridsize.x=mchan/blocksize.x+1; gridsize.y=1; gridsize.z=1;
      compute_channel_statistics<<<gridsize,blocksize>>>(mchan,mblock,msum,bs1,bs2,zavg,zstd);

      // Redigitize data to 8bits
      blocksize.x=32; blocksize.y=32; blocksize.z=1;
      gridsize.x=mchan/blocksize.x+1; gridsize.y=mblock/blocksize.y+1; gridsize.z=1;
      if (ndec==1)
	      redigitize<<<gridsize,blocksize>>>(dfbuf,mchan,mblock,msum,zavg,zstd,3.0,5.0,dcbuf);
      else
	      decimate_and_redigitize<<<gridsize,blocksize>>>(dfbuf,ndec,mchan,mblock,msum,zavg,zstd,3.0,5.0,dcbuf);      

      // Copy buffer to host
      checkCudaErrors(cudaMemcpy(cbuf,dcbuf,sizeof(unsigned char)*msamp*mchan/ndec,cudaMemcpyDeviceToHost));

      // Write buffer
      fwrite(cbuf,sizeof(char),nread*nsub/ndec,output_files[idm]);
    }
    printf("Processed %d DMs in %.2f s\n",ndm,(float) (clock()-startclock)/CLOCKS_PER_SEC);
  }

  // Close output files
  for (i=0;i<ndm;i++)
    fclose(output_files[i]);

  // Close input files
  for (i=0;i<nsub;i++)
    fclose(input_files[i]);

  // Free host memory
  free_vdif_struct(vf,nsub);
  free(vfbuf);
  free(tmp_vfbuf);
  free(dm);
  free(cbuf);
  free(output_files);
  free(input_files);
  
  // Free device memory
  cudaFree(dvfbuf);
  cudaFree(dfbuf);
  cudaFree(dcbuf);
  cudaFree(cp1);
  cudaFree(cp2);
  cudaFree(cp1p);
  cudaFree(cp2p);
  cudaFree(dc);
  cudaFree(bs1);
  cudaFree(bs2);
  cudaFree(zavg);
  cudaFree(zstd);
  cudaFree(ddm);

  // Free plan
  cufftDestroy(ftc2cf);
  cufftDestroy(ftc2cb);

  checkCudaErrors(cudaDeviceReset());

  return 0;
}

void usage()
{
  printf("CDMT - Coherent Dispersion Measure Trials\n");
  printf("Compute coherently dedispersed SIGPROC filterbank files from complex voltage data.\n\n");
  printf("Usage:\n");
  printf("  cdmt [options...] [header_file]\n\n");
  printf("Arguments:\n");
  printf("  header_file             The header file of the lowest subband\n\n");
  printf("Options:\n");
  printf("  -d <DM start,step,num>  DM start, stepsize, and number of trials\n");
  printf("  -D <GPU device>         GPU device number to use (default: 0)\n");
  printf("  -b <ndec>               Number of time samples to average (default: 1)\n");
  printf("  -N <forward FFT size>   Forward FFT size (default: 32768)\n");
  printf("  -n <overlap size>       FFT overlap size (default: 1024)\n");
  printf("  -f <FFTs per op>        Number of FFTs per cuFFT call (default: 128)\n");
  printf("  -s <nsub>               Number of input subbands (default: 24)\n");
  printf("  -c <nchan>              Channelisation factor (default: 128)\n");
  printf("  -m <msum>               Size of blocksum (default: 1024)\n");
  printf("  -o <output prefix>      Output filename prefix (default: cdmt)\n");
  printf("  -p <source name>        Source name (default: from .hdr file)\n");
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
static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int nx,int ny,int l,float scale)
{
  int i,j,k;
  i=blockIdx.x*blockDim.x+threadIdx.x;
  j=blockIdx.y*blockDim.y+threadIdx.y;

  if (i<nx && j<ny) {
    k=i+nx*j;
    c[k]=ComplexScale(ComplexMul(a[k],b[i+nx*l]),scale);
  }
}

// Compute chirp
__global__ void compute_chirp(double fcen,double bw,float *dm,int nchan,int nbin,int nsub,int ndm,cufftComplex *c)
{
  int ibin,ichan,isub,idm,mbin,idx;
  double s,rt,t,f,fsub,fchan,bwchan,bwsub;

  // Number of channels per subband
  mbin=nbin/nchan;

  // Subband bandwidth
  bwsub=bw/nsub;

  // Channel bandwidth
  bwchan=bw/(nchan*nsub);

  // Indices of input data
  isub=blockIdx.x*blockDim.x+threadIdx.x;
  ichan=blockIdx.y*blockDim.y+threadIdx.y;
  idm=blockIdx.z*blockDim.z+threadIdx.z;

  // Keep in range
  if (isub<nsub && ichan<nchan && idm<ndm) {
    // Main constant
    s=2.0*M_PI*dm[idm]/DMCONSTANT;

    // Frequencies
    fsub=fcen-0.5*bw+bw*(float) isub/(float) nsub+0.5*bw/(float) nsub;
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
      idx=ibin+ichan*mbin+isub*mbin*nchan+idm*nsub*mbin*nchan;
      
      // Chirp
      c[idx].x=cos(rt)*t;
      c[idx].y=sin(rt)*t;
    }
  }

  return;
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
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    if (isamp >= noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=4*nsamp*isub+4*(isamp-noverlap);

      cp1[idx1].x=((float)(uint8_t) dbuf[idx2]  ) - 128.0;
      cp1[idx1].y=((float)(uint8_t) dbuf[idx2+1]) - 128.0;
      cp2[idx1].x=((float)(uint8_t) dbuf[idx2+2]) - 128.0;
      cp2[idx1].y=((float)(uint8_t) dbuf[idx2+3]) - 128.0;
    }
  }

  return;
}

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are padded with noverlap samples on either side for the
// convolution. This is separate from the main kernel to minimise performance
// loss to branching on the GPU. On the first iteration, we want to fill
// the final non-noverlap region and final noverlap region so that they can 
// match the first noverlap region and first non-noverlap on the second
// iteration
__global__ void unpack_and_padd_first_iteration(char *dbuf,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft-noverlap;
    if (isamp >= noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=4*nsamp*isub+4*(isamp-noverlap);

      cp1[idx1].x=((float)(uint8_t) dbuf[idx2]  ) - 128.0;
      cp1[idx1].y=((float)(uint8_t) dbuf[idx2+1]) - 128.0;
      cp2[idx1].x=((float)(uint8_t) dbuf[idx2+2]) - 128.0;
      cp2[idx1].y=((float)(uint8_t) dbuf[idx2+3]) - 128.0;
    } else if (isamp > -noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=4*nsamp*isub+4*(noverlap-isamp);

      cp1[idx1].x=((float)(uint8_t) dbuf[idx2]  ) - 128.0;
      cp1[idx1].y=((float)(uint8_t) dbuf[idx2+1]) - 128.0;
      cp2[idx1].x=((float)(uint8_t) dbuf[idx2+2]) - 128.0;
      cp2[idx1].y=((float)(uint8_t) dbuf[idx2+3]) - 128.0;
    }
  }

  return;
}

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are located in the first noverlap region and first non-
// noverlap region, for continuous time series between data blocks
// 
// overlap_(timeblock)_(index)
// t = 0: overlap_0_0: nfft_0_0, nfft_0_1... nfft_0_N-1, nfft_0 N: overlap_0_1
// t = 1: nfft_0_N: overlap_0_1, nfft_1_0.... nfft_1_N-1:overlap_1_1
// t = 2 nfft_1_N-1: overlap_1_1...
// etc
__global__ void padd_next_iteration(char *dbuf,int nsamp,int nbin,int nfft,int nsub,int noverlap,cufftComplex *cp1,cufftComplex *cp2)
{
  int64_t ibin,ifft,isamp,isub,idx1,idx2;

  // Indices of input data
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;
  isub=blockIdx.z*blockDim.z+threadIdx.z;

  // Only compute valid threads
  if (ibin<nbin && ifft<nfft && isub<nsub) {
    isamp=ibin+(nbin-2*noverlap)*ifft;
    if (isamp<2*noverlap) {
      idx1=ibin+nbin*isub+nsub*nbin*ifft;
      idx2=4*nsamp*isub+4*(isamp+nsamp-2*noverlap);

      cp1[idx1].x=((float)(uint8_t) dbuf[idx2]  ) - 128.0;
      cp1[idx1].y=((float)(uint8_t) dbuf[idx2+1]) - 128.0;
      cp2[idx1].x=((float)(uint8_t) dbuf[idx2+2]) - 128.0;
      cp2[idx1].y=((float)(uint8_t) dbuf[idx2+3]) - 128.0;
    }
  }
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

// Compute segmented sums for later computation of offset and scale
__global__ void compute_block_sums(float *z,int nchan,int nblock,int nsum,float *bs1,float *bs2)
{
  int64_t ichan,iblock,isum,idx1,idx2;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    idx1=ichan+nchan*iblock;
    bs1[idx1]=0.0;
    bs2[idx1]=0.0;
    for (isum=0;isum<nsum;isum++) {
      idx2=ichan+nchan*(isum+iblock*nsum);
      bs1[idx1]+=z[idx2];
      bs2[idx1]+=z[idx2]*z[idx2];
    }
  }

  return;
}

// Compute segmented sums for later computation of offset and scale
__global__ void compute_channel_statistics(int nchan,int nblock,int nsum,float *bs1,float *bs2,float *zavg,float *zstd)
{
  int64_t ichan,iblock,idx1;
  double s1,s2;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  if (ichan<nchan) {
    s1=0.0;
    s2=0.0;
    for (iblock=0;iblock<nblock;iblock++) {
      idx1=ichan+nchan*iblock;
      s1+=bs1[idx1];
      s2+=bs2[idx1];
    }
    zavg[ichan]=s1/(float) (nblock*nsum);
    zstd[ichan]=s2/(float) (nblock*nsum)-zavg[ichan]*zavg[ichan];
    zstd[ichan]=sqrt(zstd[ichan]);
  }

  return;
}

// Redigitize the filterbank to 8 bits in segments
__global__ void redigitize(float *z,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz)
{
  int64_t ichan,iblock,isum,idx1;
  float zoffset,zscale;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    zoffset=zavg[ichan]-zmin*zstd[ichan];
    zscale=(zmin+zmax)*zstd[ichan];

    for (isum=0;isum<nsum;isum++) {
      idx1=ichan+nchan*(isum+iblock*nsum);
      z[idx1]-=zoffset;
      z[idx1]*=256.0/zscale;
      cz[idx1]=(unsigned char) z[idx1];
      if (z[idx1]<0.0) cz[idx1]=0;
      if (z[idx1]>255.0) cz[idx1]=255;
    }
  }

  return;
}

// Decimate and Redigitize the filterbank to 8 bits in segments
__global__ void decimate_and_redigitize(float *z,int ndec,int nchan,int nblock,int nsum,float *zavg,float *zstd,float zmin,float zmax,unsigned char *cz)
{
  int64_t ichan,iblock,isum,idx1,idx2,idec;
  float zoffset,zscale,ztmp;

  ichan=blockIdx.x*blockDim.x+threadIdx.x;
  iblock=blockIdx.y*blockDim.y+threadIdx.y;
  if (ichan<nchan && iblock<nblock) {
    zoffset=zavg[ichan]-zmin*zstd[ichan];
    zscale=(zmin+zmax)*zstd[ichan];

    for (isum=0;isum<nsum;isum+=ndec) {
      idx2=ichan+nchan*(isum/ndec+iblock*nsum/ndec);
      for (idec=0,ztmp=0.0;idec<ndec;idec++) {
	idx1=ichan+nchan*(isum+idec+iblock*nsum);
	ztmp+=z[idx1];
      }
      ztmp/=(float) ndec;
      ztmp-=zoffset;
      ztmp*=256.0/zscale;
      cz[idx2]=(unsigned char) ztmp;
      if (ztmp<0.0) cz[idx2]=0;
      if (ztmp>255.0) cz[idx2]=255;
    }
  }

  return;
}
