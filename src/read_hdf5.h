#ifndef __READ_HDF5_h
#define __READ_HDF5_h

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>

#include <hdf5.h>

// Struct for header information
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

#ifdef __cplusplus
}
#endif

#endif // __READ_HDF5_h