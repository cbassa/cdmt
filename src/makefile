# Makefile for CDMT

GPU_ARCH ?= sm_86

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
NVCC = /usr/local/cuda-12.4/bin/nvcc -arch=$(GPU_ARCH)

# Source files
CU_SRC = cdmt.cu error_check.cu
CU_HDR = cdmt.h error_check.h
C_SRC = read_hdf5.c write_sigproc.c
C_HDR = read_hdf5.h write_sigproc.h

# Object files
CU_OBJS = $(CU_SRC:.cu=.o)
C_OBJS = $(C_SRC:.c=.o)

# Executable name
EXE = cdmt

.PHONY: all clean

all: $(EXE)

$(EXE): $(CU_OBJS) $(C_OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.cu $(CU_HDR)
	$(NVCC) $(CFLAGS) -o $@ -c $<

%.o: %.c $(C_HDR)
	$(NVCC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o
	rm -f *~

cleaner: clean
	rm $(EXE)