include common/make.config

ROOT_DIR = $(shell pwd)
SRC = $(ROOT_DIR)/src
BIN = $(ROOT_DIR)/bin

all:
	cd src/sssp;	make;	mv gpu/gpu_sssp $(BIN);	rm -f gpu/*.o;	mv cpu/cpu_sssp $(BIN);	mv cpu/cpu_usssp $(BIN); rm -f cpu/*.o;
#	cd src/bc;	make;	mv bc/gpu/gpu_bc $(BIN);  rm -f *.o;
#	cd utility;	make;   mv stat $(BIN);	rm -f *.o;

clean:
	cd bin; rm -f *;
	cd src/sssp; make clean;
	cd src/bc;	make clean;
	cd utility;	make clean;                          
