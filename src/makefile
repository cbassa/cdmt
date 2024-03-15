# CDMT Makefile

# CUDA links
CUDALINK := $(shell pkg-config --libs cuda-12.4)
CUDAINCL := $(shell pkg-config --cflags cuda-12.4)

# HDF5 links and libraries
HDF5LINK := $(shell pkg-config --libs hdf5)
HDF5INCL := $(shell pkg-config --cflags hdf5)

# Compiling flags
CFLAGS = $(CUDAINCL) $(HDF5INCL)

# Linking flags
LFLAGS = -lm $(CUDALINK) -lcufft -lhdf5 -lcurand $(HDF5LINK)

# Compilers
NVCC = /usr/local/cuda-12.4/bin/nvcc

cdmt: cdmt.o
	$(NVCC) $(CFLAGS) -o cdmt cdmt.o $(LFLAGS)

cdmt.o: cdmt.cu
	$(NVCC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o
	rm -f *~
