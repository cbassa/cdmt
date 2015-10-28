# CUDA PATH
CUDAPATH = /opt/cuda

# Compiling flags
CFLAGS = -I$(CUDAPATH)/samples/common/inc

# Linking flags
LFLAGS = -lm -L$(CUDAPATH)/lib64 -lcufft -lcurand

# Compiler
NVCC = $(CUDAPATH)/bin/nvcc

cdmt: cdmt.cu
	$(NVCC) $(CFLAGS) -o cdmt cdmt.cu $(LFLAGS)

codedisp: codedisp.cu
	$(NVCC) $(CFLAGS) -o codedisp codedisp.cu $(LFLAGS)

clean:
	rm -f *.o
	rm -f *~
