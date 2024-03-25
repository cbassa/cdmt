#include "read_hdf5.h"

// This is a simple H5 reader for complex voltage data. Very little
// error checking is done.
struct header read_h5_header(char *fname)
{
  int i,len,ibeam,isap;
  struct header h;
  hid_t file_id,attr_id,sap_id,beam_id,memtype,group_id,space,coord_id;
  char *string,*pch;
  const char *stokes[]={"_S0_","_S1_","_S2_","_S3_"};
  char *froot,*fpart,*ftest,group[32];
  FILE *file;

  // Find filenames
  for (i=0;i<4;i++) {
    pch=strstr(fname,stokes[i]);
    if (pch!=NULL)
      break;
  }
  len=strlen(fname)-strlen(pch);
  froot=(char *) malloc(sizeof(char)*(len+1));
  fpart=(char *) malloc(sizeof(char)*(strlen(pch)-6));
  ftest=(char *) malloc(sizeof(char)*(len+20));
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
    h.rawfname[i]=(char *) malloc(sizeof(char)*(strlen(ftest)+1));
    strcpy(h.rawfname[i],ftest);
  }

  // Get beam number
  for (i=0;i<4;i++) {
    pch=strstr(fname,"_B");
    if (pch!=NULL)
      break;
  }
  sscanf(pch+2,"%d",&ibeam);

  // Get SAP number
  for (i=0;i<4;i++) {
    pch=strstr(fname,"_SAP");
    if (pch!=NULL)
      break;
  }
  sscanf(pch+4,"%d",&isap);

  // Free
  free(froot);
  free(fpart);
  free(ftest);

  // Open file
  file_id=H5Fopen(fname,H5F_ACC_RDONLY,H5P_DEFAULT);

  // Open subarray pointing group
  sprintf(group,"SUB_ARRAY_POINTING_%03d",isap);
  sap_id=H5Gopen(file_id,group,H5P_DEFAULT);

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

  // Open beam
  sprintf(group,"BEAM_%03d",ibeam);
  beam_id=H5Gopen(sap_id,group,H5P_DEFAULT);

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