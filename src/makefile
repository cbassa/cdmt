# Makefile for CDMT

GPU_ARCH ?= sm_86

# CUDA links
CUDALINK := $(shell pkg-config --libs cuda-12.4)
CUDAINCL := $(shell pkg-config --cflags cuda-12.4)

# VDIFIO links and libraries
VDIFIOLINK := $(shell pkg-config --libs vdifio)
VDIFIOINCL := $(shell pkg-config --cflags vdifio)

# Compiling flags
CFLAGS = $(CUDAINCL) $(VDIFIOINCL)

# Linking flags
LFLAGS = -lm $(CUDALINK) -lcufft -lcurand $(VDIFIOLINK)

# Compilers
NVCC = /usr/local/cuda-12.4/bin/nvcc -arch=$(GPU_ARCH)

# Source files
CU_SRC = cdmt.cu error_check.cu
CU_HDR = cdmt.h error_check.h
C_SRC = read_vdif.c write_sigproc.c ascii_header.c
C_HDR = read_vdif.h write_sigproc.h ascii_header.h

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