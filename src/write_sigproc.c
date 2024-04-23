#include "write_sigproc.h"

static void send_string(const char *string,FILE *file)
{
  int len;

  len=strlen(string);
  fwrite(&len,sizeof(int),1,file);
  fwrite(string,sizeof(char),len,file);

  return;
}

static void send_float(const char *string,float x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(float),1,file);

  return;
}

static void send_int(const char *string,int x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(int),1,file);

  return;
}

static void send_double(const char *string,double x,FILE *file)
{
  send_string(string,file);
  fwrite(&x,sizeof(double),1,file);

  return;
}

double ascii2double(char *ascii_coord)
{
  double double_coord;
  double deg,min,sec;  // degrees/hours, minutes, seconds
  char sign;
  char tmp[32];

  if (strlen(ascii_coord)==0)
    return 0.0;
  sscanf(ascii_coord,"%lf:%lf:%lf",&deg,&min,&sec);
  sign=(deg<0 ? '-' : ' ');
  sprintf(tmp,"%c%02d%02d%09.6lf",sign,abs((int) deg),(int) min,sec);
  sscanf(tmp,"%lf",&double_coord);
  return double_coord;
}

void write_filterbank_header(struct vdif_file *vf,FILE *file)
{
  double ra,de;

  ra=ascii2double(vf->src_raj);
  de=ascii2double(vf->src_dej);
  
  send_string("HEADER_START",file);
  send_string("rawdatafile",file);
  send_string(vf->lowdatafn,file);
  send_string("source_name",file);
  send_string(vf->src_name,file);
  send_int("machine_id",30,file);
  send_int("telescope_id",30,file);
  send_double("src_raj",ra,file);
  send_double("src_dej",de,file);
  send_int("data_type",1,file);
  send_double("fch1",vf->fch1,file);
  send_double("foff",vf->foff,file);
  send_int("nchans",vf->nchan,file);
  send_int("nbeams",0,file);
  send_int("ibeam",0,file);
  send_int("nbits",vf->nbit,file);
  send_double("tstart",vf->tstart,file);
  send_double("tsamp",vf->tsamp,file);
  send_int("nifs",1,file);
  send_string("HEADER_END",file);

  return;
}