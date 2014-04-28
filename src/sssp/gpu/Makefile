CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt 

COMMON = ../../../common
INC = ../../../common

all:	gpu_sssp

gpu_sssp:	sssp.cpp sssp_wrapper.cu sssp_kernel.cu $(COMMON)/findMin.cu $(COMMON)/findMin_kernel.cu
	$(CC) $(CC_FLAGS) -I./ -I$(INC) sssp.cpp sssp_wrapper.cu $(COMMON)/findMin.cu -O3 -o $@

clean:
	rm gpu_sssp
