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

clean:
	rm -f *.o
	rm -f *~
