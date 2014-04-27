#ifndef __SSSP_KERNEL__
#define __SSSP_KERNEL__

#include "sssp_lb_kernel.cu"

#define MAXDIMGRID 65535

#define MAX_THREAD_PER_BLOCK 1024


__global__ void O_T_B_commit_kernel(int *vertexArray, int *costArray, int *edgeArray, int *weightArray,
									char *commit, char *update, int nodeNumber, int edgeNumber, int *minValue)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && costArray[tid] == *minValue ) {
		commit[tid] = 1;
		
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			int alt = costArray[tid] + weightArray[i];
			if ( costArray[nid] > alt ) {
				//update[nid] = 1;
				atomicMin(costArray+nid, alt);		// get min cost
			}
		}
	}
}

__global__ void O_T_Q_kernel(int *vertexArray, int *costArray, int *edgeArray, int *weightArray,
									char *update, int *queue, unsigned int *qLength, 										unsigned int qMaxLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *qLength;
	if ( tid<frontierNo ){
		int curr = queue[tid];	//	grab a work from queue, bid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int i=start; i<end; ++i){
			int nid = edgeArray[i];
			int alt = costCurr + weightArray[i];
			if ( costArray[nid] > alt ) {
				//update[nid] = 1;
				atomicMin(costArray+nid, alt);		// get min cost
			}
		}
	}
}


__global__ void O_B_Q_kernel(int *vertexArray, int *costArray, int *edgeArray, int *weightArray,
									char *update, int *queue, unsigned int *qLength, unsigned int qMaxLength)
{
	int bid = blockIdx.x+blockIdx.y*gridDim.x;	// MAX_THREAD_PER_BLOCK + threadIdx.x;
	int frontierNo = *qLength;
	if ( bid<frontierNo ){
		int curr = queue[bid];	//	grab a work from queue, bid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		
		/* access neighbours */
		int tid = threadIdx.x;		
		int stride = (end-start)/blockDim.x + 1; 
		for (int i=0; i<stride; i++){
			if ( tid<end-start) {
				int nid = edgeArray[start+tid];
				int alt = costArray[curr] + weightArray[start+tid];
				if ( costArray[nid] > alt ) {
					update[nid] = 1;
					atomicMin(costArray+nid, alt);		// get min cost
				}
			}
			tid += blockDim.x;
		}
	}
}

__global__ void O_B_B_commit_kernel(int *vertexArray, int *costArray, int *edgeArray, int *weightArray,
									char *commit, char *update, int nodeNumber, int edgeNumber, int *minValue)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	
	if ( bid<nodeNumber && costArray[bid] == *minValue ) {
		commit[bid] = 1;
		
		/* get neighbour range */
		int start = vertexArray[bid];
		int end = vertexArray[bid+1];
		/* access neighbours */		
		int tid = threadIdx.x;
		int stride = (end-start)/blockDim.x + 1;
		int costCurr = costArray[bid];
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

__global__ void unorder_threadBitmap_kernel(	int *vertexArray, int *costArray, int *edgeArray, int *weightArray, 
												char *frontier, char *update, int nodeNumber)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && frontier[tid] ) {
		frontier[tid] = 0;
		/* get neighbour range */				
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		/* access neighbours */
		int costCurr = costArray[tid];
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			int alt = costCurr + weightArray[i];
			if ( costArray[nid] > alt ) {
				atomicMin(costArray+nid, alt);
				update[nid] = 1;	// update neighbour needed
			}
		}
	}
}

__global__ void unorder_blockBitmap_kernel( int *vertexArray, int *costArray, int *edgeArray, int *weightArray, 
									char *frontier, char *update, int nodeNumber )
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	if ( bid<nodeNumber && frontier[bid] ) {
		/* get neighbour range */				
		int start = vertexArray[bid];
		int end = vertexArray[bid+1];
		/* access neighbours */		
		int tid = threadIdx.x;
		int stride = (end-start)/blockDim.x + 1;
		int costCurr = costArray[bid];
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

__global__ void unorder_threadQueue_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue,unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *qLength;
	if ( tid<frontierNo ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
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
}

__global__ void unorder_blockQueue_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue, unsigned int *qLength)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x; //*MAX_THREAD_PER_BLOCK + threadIdx.x;	
	int frontierNo = *qLength;
	if ( bid<frontierNo ) {
		int curr = queue[bid];	//	grab a work from queue, tid is queue index
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

__global__ void countWorkingset_kernel( char *update, unsigned int *qCounter, 
										unsigned int qMaxLength, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && update[tid] )
		atomicInc(qCounter, qMaxLength);
}

__global__ void generateBitmap_kernel( char *frontier, char *update, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && update[tid] ) {
		frontier[tid] = 1;
	}
}

__global__ void order_generateQueue_kernel(	int *costArray, int *queue, char *commit, unsigned int *qCounter, 
											unsigned int qMaxLength, int nodeNumber, int *minValue)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && costArray[tid] == *minValue ) {
		commit[tid] = 1;
		unsigned int qIndex = atomicInc(qCounter, qMaxLength);
		queue[qIndex] = tid;
	}
}

__global__ void unorder_generateQueue_kernel(	char *update, int nodeNumber, int *queue, 
												unsigned int *qCounter, unsigned int qMaxLength)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		/* write node number to queue */
		unsigned int qIndex = atomicInc(qCounter, qMaxLength);
		queue[qIndex] = tid;
	}
}
#endif
