#define THREASHOLD 100;

__global__ void unorder_threadQueue_lb_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue,unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *qLength;
	// 1st phase
	if ( tid<frontierNo ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edgeNum = end - start;
		if ( edgeNum<THREASHOLD ) {
			/* access neighbours */
			int costCurr = costArray[curr];
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				int alt = costCurr + weightArray[i];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
		else {


		}
	}
	__syncthreads();
	// 2nd phase, try block mapping
//	for ();
}


/**************************************************************************
**  Priority queue require both compute kernel and queue generation kernel
***************************************************************************/
// /__global__