CC = g++
FLAGS =

COMMON = ../../../common
INC = -I./ -I../../../common

all:	cpu_sssp cpu_usssp

cpu_sssp:	sssp.cpp sssp.h  $(COMMON)/heap.cpp  $(COMMON)/heap.h
	${CC} ${FLAGS} $(INC) sssp.cpp sssp.h  $(COMMON)/heap.cpp -o $@
cpu_usssp:	usssp.cpp usssp.h
	${CC} ${FLAGS} $(INC) usssp.cpp usssp.h -o $@
clean:
	rm cpu_sssp cpu_usssp
