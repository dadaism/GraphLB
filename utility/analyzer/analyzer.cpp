#include "analyzer.h"

using namespace std;

struct _INFO_ info;

int main()
{	
    //readInputDIMACS9();
	//readInputDIMACS10();
	readInputSLNDC();	
	getStatInfo();
	return 0;
}

void getStatInfo()
{
	// total number of nodes and edges
	info.noNodeTotal = noNodeTotal;
	info.noEdgeTotal = noEdgeTotal;

	int maxDegreeNodeId;
	int minDegreeNodeId;
	int *nodeDegree = new int [noNodeTotal];
	info.avgDegree = 0;
	info.minDegree = 999999;
	info.maxDegree = 0;
	/* get "avg" "min" "max" of degree */
	for (int i=0; i<noNodeTotal; ++i){
		int degree = 0;
		//printf("Size of node %d is %d\n", i, adjacencyNodeList[i].size() );
		while ( adjacencyNodeList[i].empty()!=true &&
				adjacencyWeightList[i].empty()!=true){
			//adjacencyNodeList[i].back();
			//adjacencyWeightList[i].back();
			degree++;
			adjacencyNodeList[i].pop_back();
			adjacencyWeightList[i].pop_back();
		}
		//if (degree==0){
		//	printf("For node %d, degree is %d\n", i, degree);
		//}
		nodeDegree[i] = degree;
		info.totalDegree += degree;
		if ( degree>info.maxDegree ){
			info.maxDegree = degree;
			maxDegreeNodeId = i;		
		}
		if ( degree<info.minDegree ){
			info.minDegree = degree;
			minDegreeNodeId = i;		
		}
	}
	info.avgDegree = (float)info.totalDegree / info.noNodeTotal;
	printf("Node:%d\n", info.noNodeTotal);
	printf("Edge by degree:%ld\n", info.totalDegree/2);
	printf("Edge:%d\n", info.noEdgeTotal);
	printf("Avg:\t%f\nMax:\t%d\nMin:\t%d\n",info.avgDegree, info.maxDegree, info.minDegree);
	printf("Node ID with max degree:%d\n", maxDegreeNodeId);
	printf("Node ID with min degree:%d\n", minDegreeNodeId);
	/* get distribution of degree */
	info.distDegree = new long [ info.maxDegree + 1];
	for (int i=info.minDegree; i<=info.maxDegree; ++i)
		info.distDegree[i] = 0;

	for (int i=0; i<noNodeTotal; ++i){
		info.distDegree[ nodeDegree[i] ]++;
	}
	
	/* print distribution of degree */
	/*for (int i=info.minDegree; i<=info.maxDegree; ++i){
		if ( info.distDegree[i]!=0 )
			printf("%d %d\n", i, info.distDegree[i]);
	}*/
	delete [] nodeDegree;
}
