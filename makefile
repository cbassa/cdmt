# CUDA PATH
CUDAPATH = /opt/cuda

# Compiling flags
CFLAGS = -I$(CUDAPATH)/samples/common/inc

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5

# Compilers
NVCC = $(CUDAPATH)/bin/nvcc
CC = gcc

cdmt: cdmt.o
	$(NVCC) $(CFLAGS) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

codedisp: codedisp.cu
	$(NVCC) $(CFLAGS) -o codedisp codedisp.cu $(LFLAGS)

cdmt_rcvr: cdmt_rcvr.c
	$(CC) -o cdmt_rcvr cdmt_rcvr.c

cdmt_join: cdmt_join.cu
	$(NVCC) $(CFLAGS) -o cdmt_join cdmt_join.cu -lm $(LFLAGS)

#cdmt_join: cdmt_join.c
#	$(CC) -o cdmt_join cdmt_join.c -lm


clean:
	rm -f *.o
	rm -f *~
