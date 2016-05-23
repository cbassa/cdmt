#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <getopt.h>

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
  printf("cdmt_join -o <outfile> -n [decimate] <input files in decreasing frequency order>\n");

  return;
}

// Decimate the timeseries in time. Reuses input array                                     
void decimate_timeseries(unsigned char *z,int nx,int ny,int mx)
{
  int64_t i,j,k,l;
  float ztmp;
  unsigned char c;

  for (j=0,l=0;j<ny;j+=mx,l++) {
    for (i=0;i<nx;i++) {
      ztmp=0.0;
      for (k=0;k<mx;k++)
        ztmp+=(float) z[i+nx*(j+k)];
      ztmp/=mx;
      if (ztmp>255.0)
        c=(unsigned char) 255;
      else if (ztmp<0.0)
        c=(unsigned char) 0;
      else
        c=(unsigned char) ztmp;
      z[i+nx*l]=c;
    }
  }

  return;
}

int main(int argc,char *argv[])
{
  int i,nchan,nread,nfile,ndec=1;
  FILE **file,*ofile;
  char *ofname;
  struct header *h;
  unsigned char *buffer;
  int arg=0;

  // Decode options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"o:hn:"))!=-1) {
      switch(arg) {

      case 'o':
	ofname=optarg;
	break;

      case 'n':
	ndec=atoi(optarg);
	break;

      case 'h':
	usage();
	return 0;

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
  nfile=argc-optind;

  //Open output file
  ofile=fopen(ofname,"w");
  if (ofile==NULL) {
    fprintf(stderr,"Failed to open file %s\n",ofname);
    return -1;
  }

  // Allocate files
  file=(FILE **) malloc(sizeof(FILE *)*nfile);

  // Allocate headers
  h=(struct header *) malloc(sizeof(struct header)*nfile);

  // Open files and read headers
  printf("File                     BW (MHz) Fch1 (MHz) Tobs (s) MJDstart\n");
  for (i=0;i<nfile;i++) {
    file[i]=fopen(argv[i+optind],"r");
    if (file[i]==NULL) {
      fprintf(stderr,"Failed to open file %s\n",argv[i+optind]);
      return -1;
    }
    h[i]=read_header(file[i]);
    printf("%s; %f %f %.3f %.8lf\n",argv[i+optind],fabs(h[i].nchan*h[i].foff),h[i].fch1,h[i].nsamp*h[i].tsamp,h[i].tstart);
  }

  // Perform checks here

  // Allocate buffer
  buffer=(unsigned char *) malloc(sizeof(unsigned char)*h[0].nchan); 

  // Adjust and write header
  nchan=h[0].nchan;
  h[0].nchan*=nfile;
  h[0].tsamp*=ndec;
  write_header(h[0],ofile);

  // Loop over files and spectra
  for (;;) {
    for (i=0;i<nfile;i++) {
      nread=fread(buffer,sizeof(unsigned char),nchan*ndec,file[i]);
      if (nread==0)
	break;
      if (ndec>1)
	decimate_timeseries(buffer,nchan,ndec,ndec);
      fwrite(buffer,sizeof(unsigned char),nchan,ofile);
    }
    if (nread==0)
      break;
  }

  // Close files
  for (i=0;i<nfile;i++)
    fclose(file[i]);
  fclose(ofile);

  // Free
  free(file);
  free(h);
  free(buffer);

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
