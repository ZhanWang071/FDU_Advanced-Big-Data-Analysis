
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
# NVCCFLAGS	:= -lineinfo -arch=sm_50 --ptxas-options=-v --use_fast_math

all:	pointer addition_cpu addition_gpu

pointer:	pointer.c Makefile
	gcc -Wall -std=c99 -o pointer pointer.c

addition_cpu:	addition_cpu.cu Makefile
	nvcc addition_cpu.cu -o addition_cpu $(INC) $(NVCCFLAGS) $(LIB)

addition_gpu:	addition_gpu.cu Makefile
	nvcc addition_gpu.cu -o addition_gpu $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f pointer addition_cpu addition_gpu
