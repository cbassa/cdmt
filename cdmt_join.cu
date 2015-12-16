#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>


struct header {
  int64_t headersize,buffersize;
  int nchan,nsamp,nbit,nif;
  int machine_id,telescope_id,nbeam,ibeam,sumif;
  double tstart,tsamp,fch1,foff;
  double src_raj,src_dej,az_start,za_start;
  char source_name[80],ifstream[8],inpfile[80];
};
struct header read_header(FILE *file);
void write_header(struct header h,FILE *file);

void usage(void)
{
  printf("cdmt_join -o <outfile> <input files in decreasing frequency order>\n");

  return;
}

// Reordering kernel
__global__ void reorder(unsigned char *obuf,unsigned char *ibuf,int64_t nsamp,int64_t nchan,int64_t ipart,int64_t npart)
{
  int64_t isamp,ichan,idx1,idx2;

  // Indices of input data
  isamp=blockIdx.x*blockDim.x+threadIdx.x;
  ichan=blockIdx.y*blockDim.y+threadIdx.y;

  // Compute valid threads
  if (isamp<nsamp && ichan<nchan) {
    idx1=ichan+isamp*nchan;
    idx2=ichan+ipart*nchan+isamp*npart*nchan;
    obuf[idx2]=ibuf[idx1];
  }

  return;
}


int main(int argc,char *argv[])
{
  int64_t ipart,npart,nsamp=5000000,nsampread,nchan;
  FILE **file,*ofile;
  char *ofname;
  struct header *h;
  unsigned char *ibuf,*obuf;
  unsigned char *dibuf,*dobuf;
  clock_t startclock;
  int arg=0,device=0;
  dim3 blocksize,gridsize;
  float tread=0.0,treorder=0.0,twrite=0.0,tcopy=0.0;

  // Decode options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"o:hd:"))!=-1) {
      switch(arg) {

      case 'o':
	ofname=optarg;
	break;

      case 'h':
	usage();
	return 0;

      case 'd':
	device=atoi(optarg);
	break;

      default:
	usage();
	return 0;
      }
    }
  } else {
    usage();
    return 0;
  }
  
  // Get number of files
  npart=argc-optind;

  //Open output file
  ofile=fopen(ofname,"w");
  if (ofile==NULL) {
    fprintf(stderr,"Failed to open file %s\n",ofname);
    return -1;
  }

  // Allocate files
  file=(FILE **) malloc(sizeof(FILE *)*npart);

  // Allocate headers
  h=(struct header *) malloc(sizeof(struct header)*npart);

  // Open files and read headers
  printf("File                     BW (MHz) Fch1 (MHz) Tobs (s) MJDstart\n");
  for (ipart=0;ipart<npart;ipart++) {
    file[ipart]=fopen(argv[ipart+optind],"r");
    if (file[ipart]==NULL) {
      fprintf(stderr,"Failed to open file %s\n",argv[ipart+optind]);
      return -1;
    }
    h[ipart]=read_header(file[ipart]);
    printf("%s; %f %f %.3f %.8lf\n",argv[ipart+optind],fabs(h[ipart].nchan*h[ipart].foff),h[ipart].fch1,h[ipart].nsamp*h[ipart].tsamp,h[ipart].tstart);
  }

  // Perform checks here

  // Adjust and write header
  nchan=h[0].nchan;
  h[0].nchan*=npart;
  write_header(h[0],ofile);

  // Set device                                                                            
  checkCudaErrors(cudaSetDevice(device));

  // Allocate buffers
  ibuf=(unsigned char *) malloc(sizeof(unsigned char)*nchan*nsamp); 
  obuf=(unsigned char *) malloc(sizeof(unsigned char)*nchan*nsamp*npart); 
  checkCudaErrors(cudaMalloc((void **) &dibuf,sizeof(unsigned char)*nchan*nsamp));
  checkCudaErrors(cudaMalloc((void **) &dobuf,sizeof(unsigned char)*nchan*nsamp*npart));

  // Loop over blocks
  for (;;) {
    // Loop over files
    for (ipart=0;ipart<npart;ipart++) {
      startclock=clock();
      nsampread=fread(ibuf,sizeof(unsigned char),nchan*nsamp,file[ipart])/nchan;
      tread+=(float) (clock()-startclock)/CLOCKS_PER_SEC;

      // Break if file empty
      if (nsampread==0)
	break;

      // Copy input buffer to GPU
      startclock=clock();
      checkCudaErrors(cudaMemcpy(dibuf,ibuf,sizeof(unsigned char)*nchan*nsampread,cudaMemcpyHostToDevice));
      tcopy+=(float) (clock()-startclock)/CLOCKS_PER_SEC;

      // Reorder buffer
      blocksize.x=256;blocksize.y=4;blocksize.z=1;
      gridsize.x=nsampread/blocksize.x+1;gridsize.y=nchan/blocksize.y+1;gridsize.z=1;
      startclock=clock();
      reorder<<<gridsize,blocksize>>>(dobuf,dibuf,nsampread,nchan,ipart,npart);
      checkCudaErrors(cudaDeviceSynchronize());
      treorder+=(float) (clock()-startclock)/CLOCKS_PER_SEC;
    }
    // Break if file empty
    if (nsampread==0)
      break;

    // Copy output buffer to host
    startclock=clock();
    checkCudaErrors(cudaMemcpy(obuf,dobuf,sizeof(unsigned char)*nchan*nsampread*npart,cudaMemcpyDeviceToHost));
    tcopy+=(float) (clock()-startclock)/CLOCKS_PER_SEC;

    // Write buffer
    startclock=clock();
    fwrite(obuf,sizeof(unsigned char),nchan*npart*nsampread,ofile);
    twrite+=(float) (clock()-startclock)/CLOCKS_PER_SEC;
  }
  printf("Reading %.2f s, writing: %.2f s, GPU copy, %.2f s, GPU process: %.2f s\n",tread,twrite,tcopy,treorder);

  // Close files
  for (ipart=0;ipart<npart;ipart++)
    fclose(file[ipart]);
  fclose(ofile);

  // Free
  free(file);
  free(h);
  free(ibuf);
  free(obuf);
  cudaFree(dobuf);
  cudaFree(dibuf);

  return 0;
}

// Read SIGPROC filterbank header
struct header read_header(FILE *file)
{
  int nchar,nbytes=0;
  char string[80];
  struct header h;

  // Read header parameters
  for (;;) {
    // Read string size
    strcpy(string,"ERROR");
    fread(&nchar,sizeof(int),1,file);
    
    // Skip wrong strings
    if (!(nchar>1 && nchar<80)) 
      continue;

    // Increate byte counter
    nbytes+=nchar;

    // Read string
    fread(string,nchar,1,file);
    string[nchar]='\0';
    
    // Exit at end of header
    if (strcmp(string,"HEADER_END")==0)
      break;
    
    // Read parameters
    if (strcmp(string, "tsamp")==0) 
      fread(&h.tsamp,sizeof(double),1,file);
    else if (strcmp(string,"tstart")==0) 
      fread(&h.tstart,sizeof(double),1,file);
    else if (strcmp(string,"fch1")==0) 
      fread(&h.fch1,sizeof(double),1,file);
    else if (strcmp(string,"foff")==0) 
      fread(&h.foff,sizeof(double),1,file);
    else if (strcmp(string,"nchans")==0) 
      fread(&h.nchan,sizeof(int),1,file);
    else if (strcmp(string,"nifs")==0) 
      fread(&h.nif,sizeof(int),1,file);
    else if (strcmp(string,"nbits")==0) 
      fread(&h.nbit,sizeof(int),1,file);
    else if (strcmp(string,"nsamples")==0) 
      fread(&h.nsamp,sizeof(int),1,file);
    else if (strcmp(string,"az_start")==0) 
      fread(&h.az_start,sizeof(double),1,file);
    else if (strcmp(string,"za_start")==0) 
      fread(&h.za_start,sizeof(double),1,file);
    else if (strcmp(string,"src_raj")==0) 
      fread(&h.src_raj,sizeof(double),1,file);
    else if (strcmp(string,"src_dej")==0) 
      fread(&h.src_dej,sizeof(double),1,file);
    else if (strcmp(string,"telescope_id")==0) 
      fread(&h.telescope_id,sizeof(int),1,file);
    else if (strcmp(string,"machine_id")==0) 
      fread(&h.machine_id,sizeof(int),1,file);
    else if (strcmp(string,"nbeams")==0) 
      fread(&h.nbeam,sizeof(int),1,file);
    else if (strcmp(string,"ibeam")==0) 
      fread(&h.ibeam,sizeof(int),1,file);
    else if (strcmp(string,"source_name")==0) 
      strcpy(h.source_name,string);
  }

  // Get header and buffer sizes
  h.headersize=(int64_t) ftell(file);
  fseek(file,0,SEEK_END);
  h.buffersize=ftell(file)-h.headersize;
  h.nsamp=h.buffersize/(h.nchan*h.nif*h.nbit/8);

  // Reset file pointer to start of buffer
  rewind(file);
  fseek(file,h.headersize,SEEK_SET);

  return h;
}

void send_string(const char *string,FILE *file)
{
  int len;

  len=strlen(string);
  fwrite(&len,sizeof(int),1,file);
  fwrite(string,sizeof(char),len,file);

  return;
}

void send_float(const char *string,float x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(float),1,file);

  return;
}

void send_int(const char *string,int x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(int),1,file);

  return;
}

void send_double(const char *string,double x,FILE *file)
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

void write_header(struct header h,FILE *file)
{
  
  send_string("HEADER_START",file);
  send_string("rawdatafile",file);
  send_string("/somedir/somefile.fil",file);
  send_string("source_name",file);
  send_string(h.source_name,file);
  send_int("machine_id",11,file);
  send_int("telescope_id",11,file);
  send_double("src_raj",h.src_raj,file);
  send_double("src_dej",h.src_dej,file);
  send_int("data_type",1,file);
  send_double("fch1",h.fch1,file);
  send_double("foff",h.foff,file);
  send_int("nchans",h.nchan,file);
  send_int("nbeams",0,file);
  send_int("ibeam",0,file);
  send_int("nbits",h.nbit,file);
  send_double("tstart",h.tstart,file);
  send_double("tsamp",h.tsamp,file);
  send_int("nifs",1,file);
  send_string("HEADER_END",file);

  return;
}
