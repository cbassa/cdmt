#ifndef __WRITE_SIGPROC_h
#define __WRITE_SIGPROC_h

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "read_hdf5.h"

static void send_string(const char *string,FILE *file);
static void send_float(const char *string,float x,FILE *file);
static void send_int(const char *string,int x,FILE *file);
static void send_double(const char *string,double x,FILE *file);
static double dec2sex(double x);
void write_filterbank_header(struct header h,FILE *file);

#ifdef __cplusplus
}
#endif

#endif // __WRITE_SIGPROC_h