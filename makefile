# CUDA PATH
CUDAPATH = /opt/cuda

# Compiling flags
CFLAGS = -I$(CUDAPATH)/samples/common/inc

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lhdf5 -lcurand

# Compilers
NVCC = $(CUDAPATH)/bin/nvcc
CC = gcc

cdmt: cdmt.o
	$(NVCC) $(CFLAGS) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

cdmt_bench: cdmt_bench.o
	$(NVCC) $(CFLAGS) -o cdmt_bench cdmt_bench.o $(LFLAGS)

cdmt_bench.o: cdmt_bench.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

cdmt_disk: cdmt_disk.o
	$(NVCC) $(CFLAGS) -o cdmt_disk cdmt_disk.o $(LFLAGS)

cdmt_disk.o: cdmt_disk.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

codedisp: codedisp.cu
	$(NVCC) $(CFLAGS) -o codedisp codedisp.cu $(LFLAGS)

cdmt_rcvr: cdmt_rcvr.c
	$(CC) -o cdmt_rcvr cdmt_rcvr.c

#cdmt_join: cdmt_join.cu
#	$(NVCC) $(CFLAGS) -o cdmt_join cdmt_join.cu -lm $(LFLAGS)

cdmt_join: cdmt_join.c
	$(CC) -o cdmt_join cdmt_join.c -lm


clean:
	rm -f *.o
	rm -f *~
