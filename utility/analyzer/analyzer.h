#ifndef __ANALYZER_H__
#define __ANALYZER_H__

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "../shared/global.h"

struct _INFO_ {
	int noNodeTotal;
	int noEdgeTotal;
	long totalDegree;
	float avgDegree;
	int minDegree;
	int maxDegree;
	long *distDegree;
};

void getStatInfo();

#endif
