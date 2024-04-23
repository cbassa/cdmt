#ifndef __READ_VDIF_h
#define __READ_VDIF_h

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include <vdifio.h>

#include "ascii_header.h"

#define FPS 10000  // Frames per second as per VCSBeam
#define VDIF_FRAME_BYTES 544
#define VDIF_DATA_BYTES VDIF_FRAME_BYTES-VDIF_HEADER_BYTES

struct vdif_file {
  char **hdrfns;       // nsub header file names
  char **datafns;      // nsub data file names
  char *lowdatafn;     // lowest freq data file; to store in filterbank header
  char *telescope;     // telescope name
  char *src_name;      // source name
  char *src_raj;       // source right ascension
  char *src_dej;       // source declination
  unsigned int nsub;   // num of subbands
  unsigned int nchan;  // channelisation factor
  unsigned int nbit;   // number of bits
  double fcen;         // centre of full band
  double bwchan;       // subband bandwidth
  double tsamp;        // native sampling time
  double tstart;       // start time
  double fch1;         // centre of highest-freq fine channel
  double foff;         // width of fine channel
};

struct vdif_file *init_vdif_struct(unsigned int nsub);
void free_vdif_struct(struct vdif_file *vf, unsigned int nsub);
int parse_ascii_header(struct vdif_file *vf,unsigned int nsub,char *hdrfn,char *src_name);
int get_lowest_subband(char *hdrfile);
char *get_common_fname(char *hdrfile);
double get_start_mjd(char *datafn);
int read_block_and_strip(FILE *file,char *vfbuf,char *tmp_vfbuf,int isub,int nframe);

#ifdef __cplusplus
}
#endif

#endif // __READ_VDIF_h