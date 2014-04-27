#include "sssp.h"

using namespace std;

unsigned int data_set_num = 0;
unsigned int solution = 0;
unsigned int device_num = 0;

int main(int argc, char* argv[])
{
	double time, end_time;

	if ( argc==4 ) {
        data_set_num = atoi(argv[1]);
        solution = atoi(argv[2]);
        device_num = atoi(argv[3]);
    }
	else {
        printf("Usage: sssp [dataset] [solution] [deviceNumber] < /PATH TO DATA FILE\n");
        printf("dataset: 0 - DIMACS9\n");
        printf("         1 - DIMACS10\n");
        printf("         2 - SLNDC\n");
        printf("solution: 0 - Ordered + thread bitmap\n");
        printf("          1 - Ordered + thread queue\n");
        printf("          2 - Ordered + block bitmap\n");
        printf("          3 - Ordered + block queue\n");
        printf("          4 - Unordered + thread bitmap\n");
        printf("          5 - Unordered + thread queue\n");
        printf("          6 - Unordered + block bitmap\n");
        printf("          7 - Unordered + block queue\n");
        printf("          8 - Unordered + thread mapping + queue + delayed buffer\n");
        printf("          9 - Unordered + thread mapping + priority queue\n");
        printf("device number: start from 0\n");
        exit(0);
    }

	
	time = gettime();
	switch(data_set_num) {
        case 0: readInputDIMACS9(); break;
        case 1: readInputDIMACS10(); break;
        case 2: readInputSLNDC(); break;
        default: printf("Wrong code for dataset\n"); break;
    }

	printf("Source node is: %d\n", source+1);
	end_time = gettime();
	printf("Read data:\t\t%lf\n",end_time-time);
	
	time = gettime();
	convertCSR();
	end_time = gettime();
	printf("AdjList to CSR:\t\t%lf\n",end_time-time);

	/* initialize the unordered working set */
	queue<int> myqueue;
	SSSP_queue_init(myqueue, 0, 0);
	SSSP_GPU(	graph.vertexArray, graph.edgeArray, graph.weightArray, 
				graph.costArray, graph.frontier, noNodeTotal, 
				noEdgeTotal, source );
	//outputCost();
	delete [] graph.vertexArray;
	delete [] graph.edgeArray;
	delete [] graph.weightArray;
	delete [] graph.frontier;
	//delete [] graph.visited;

	return 0;
}
