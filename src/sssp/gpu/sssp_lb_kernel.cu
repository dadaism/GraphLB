#ifndef __SSSP_LB_KERNEL__
#define __SSSP_LB_KERNEL__

#define THREASHOLD 100
#define BUFF_SIZE 50

__global__ void unorder_threadQueue_lb_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue,unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *qLength;
	int t_idx = 0;
	__shared__ unsigned int idx;
	__shared__ int buffer[BUFF_SIZE];
	if (tid==0)
		idx = 0;
	__syncthreads();
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
		else { // insert into delayed buffer
			t_idx = atomicInc(&idx, BUFF_SIZE);
			buffer[t_idx] = queue[tid];
		}
	}
	__syncthreads();
	// 2nd phase, try block mapping
	for (int i=0; i<t_idx; ++i) {
		int curr = buffer[i];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int tid = threadIdx.x;
		int stride = (end-start)/blockDim.x + 1;
		int costCurr = costArray[curr];
		for (int i=0; i<stride; i++){
			if ( tid<end-start ){
				int eid = start+tid;	// edge id
				int nid = edgeArray[eid];	// neighbour id
				int alt = costCurr + weightArray[eid];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);				
					update[nid] = 1;	// update neighbour needed
				}
			}
			tid += blockDim.x;
		}
	}
}


/** @brief Priority queue require new queue generation kernel
*          The compute kernel can be reused from threadQueue or blockQueue
*   @param queue_0 queue for vertice with low degree
*   @param queue_1 queue for vertice with high degree
*/
__global__ void unorder_generateQueue_lb_kernel(	int *vertexArray, char *update, int nodeNumber, 
													int *queue_0, unsigned int *qCounter_0, unsigned int qMaxLength_0,
													int *queue_1, unsigned int *qCounter_1, unsigned int qMaxLength_1)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edgeNum = end - start;
		if ( edgeNum<THREASHOLD ) {
			/* write vertex number to LOW degree queue */
			unsigned int qIndex = atomicInc(qCounter_0, qMaxLength_0);
			queue_0[qIndex] = tid;
		}
		else {
			/* write vertex number to HIGH degree queue */
			unsigned int qIndex = atomicInc(qCounter_1, qMaxLength_1);
			queue_1[qIndex] = tid;
		}
	}
}

#define LIMIT 100 
#define NESTED_BLOCK_SIZE 128 
#define DEBUG

/** @brief neighbor processing kernel launched by dynamic parallelism
*   @param edgeArray
*   @param weightArrat
*   @param start starting index in edgeArray
*   @param end ending index in edgeArray
*/
__global__ void sssp_process_neighbors(	int *edgeArray, int *weightArray, int *costArray, 
										char *update, int costCurr, int start, int end)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
	if (tid < end) {
       	int nid = edgeArray[tid];
		int alt = costCurr + weightArray[tid];
		if ( costArray[nid] > alt ) {
			atomicMin(costArray+nid, alt);
			update[nid] = 1;	// update neighbour needed
		}
	}
}

__global__ void unorder_threadQueue_dp_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
												char *update, int nodeNumber, int *queue,unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *qLength;
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
#ifdef DEBUG
			printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
			int costCurr = costArray[curr];
			sssp_process_neighbors<<<edgeNum/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE>>>(
									edgeArray, weightArray, costArray, update, costCurr, start, end);

		}
	}
}

#endif