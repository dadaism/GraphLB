#ifndef __SSSP_H__
#define __SSSP_H__

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "global.h"

void SSSP_GPU(	int *vertexArray, int *edgeArray, int *levelArray,
			 	int *cost, char *frontier, int nodeNumber, 
				int edgeNumber, int source );

#endif
