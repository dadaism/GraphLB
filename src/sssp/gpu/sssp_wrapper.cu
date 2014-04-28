#include <stdio.h>
#include <cuda.h>

#include "sssp_kernel.cu"
#include "findMin.h"

#define INF 1073741824	// 1024*1024*1024
#define QMAXLENGTH 10240000

extern unsigned int solution;
extern unsigned int device_num;

inline void cudaCheckError(int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
		exit(1);
	}
}

extern double gettime();

void SSSP_GPU(  int *vertexArray, int *edgeArray, int *weightArray,
				int *costArray, char *frontier, int nodeNumber, 
				int edgeNumber, int source)
{
	int noPerBlock = nodeNumber;
	int minValue = 0;

	int *d_vertexArray;
	int *d_costArray;	
	int *d_edgeArray;
	int *d_weightArray;
	int *d_workQueue;
	int *d_workQueue_1;
	int *d_bufferBlock_1024_1024;
	int *d_bufferBlock_1024;
	int *d_minValue;
	char *d_frontier;
	char *d_update;
	char *commit = new char [nodeNumber]();
	char *update = new char [nodeNumber]();	
	char *d_commit;
	unsigned int *d_qCounter;
	unsigned int *d_qLength;
	unsigned int *d_qLength_1;

	dim3 dimGrid(1,1,1);	// thread+bitmap
	dim3 dimBlock(1,1,1);
	int maxDegreeB = 32;	
	dim3 dimBGrid(1,1,1);	// block+bitmap
	dim3 dimBBlock(maxDegreeB,1,1);
	int maxDegreeT = 192;	// thread/block, thread+queue
	dim3 dimGridT(1,1,1);
	dim3 dimBlockT(maxDegreeT,1,1);

	dim3 dimGridB(1,1,1);	// block+queue
	dim3 dimBlockB(maxDegreeB,1,1);

	unsigned int qMaxLength = QMAXLENGTH;
	int *workQueue = new int [qMaxLength];
	unsigned int qLength = 0;
	unsigned int qCounter = 0;

	unsigned int qMaxLength_1 = QMAXLENGTH / 5;
	int *workQueue_1 = new int [qMaxLength_1];
	unsigned int qLength_1 = 0;

	double time, end_time;
	time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(device_num) );
	printf("Choose CUDA device: %d\n", device_num);
	end_time = gettime();
	printf("cudaSetDevice:\t\t%lf\n",end_time-time);

	if ( noPerBlock > maxDegreeT ){
		dimGrid.x = nodeNumber / maxDegreeT + 1;
		dimBlock.x = maxDegreeT;
	}
	else {
		dimGrid.x = 1;
		dimBlock.x = noPerBlock;
	}
	/* Configuration for block+bitmap */
	if ( nodeNumber > MAXDIMGRID ){
		dimBGrid.x = MAXDIMGRID;
		dimBGrid.y = nodeNumber / MAXDIMGRID + 1;
	}
	else {
		dimBGrid.x = nodeNumber;
	}

	/* initialization */
	for (int i=0; i<nodeNumber; i++ ) {
		costArray[i] = INF;
	}
	update[source] = 1;
	costArray[source] = 0;
	
	//printf("Active number in queue:%d\n", qLength);

	/* Allocate GPU memory */
	time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(nodeNumber+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_costArray, sizeof(int)*nodeNumber ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*edgeNumber ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_weightArray, sizeof(int)*edgeNumber ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_frontier, sizeof(char)*nodeNumber ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_update, sizeof(char)*nodeNumber ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_commit, sizeof(char)*nodeNumber ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_workQueue, sizeof(int)*qMaxLength) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_qCounter, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_qLength, sizeof(unsigned int) ) );
	if ( solution<4 ) {
		cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bufferBlock_1024, sizeof(int)*1024 ) );
		cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bufferBlock_1024_1024, sizeof(int)*1024*1024 ) );	
	}
	if ( solution==9 ) {
		cudaCheckError( __LINE__, cudaMalloc( (void**)&d_workQueue_1, sizeof(int)*qMaxLength_1) );
		cudaCheckError( __LINE__, cudaMalloc( (void**)&d_qLength_1, sizeof(unsigned int) ) );
	}
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_minValue, sizeof(int)) );
	end_time = gettime();
	printf("cudaMalloc:\t\t%lf\n",end_time-time);

	time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, vertexArray, sizeof(int)*(nodeNumber+1), cudaMemcpyHostToDevice) );		
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, edgeArray, sizeof(int)*edgeNumber, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_costArray, costArray, sizeof(int)*nodeNumber,cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_weightArray, weightArray, sizeof(int)*edgeNumber, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_update, update,  sizeof(char)*nodeNumber, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_qCounter, &qCounter, sizeof(unsigned int), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_qLength, &qLength, sizeof(unsigned int), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_minValue, &minValue, sizeof(int), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemset(d_commit, 0, sizeof(char)*nodeNumber) );
	if ( solution==9 ) {
		cudaCheckError( __LINE__, cudaMemcpy( d_qLength_1, &qLength_1, sizeof(unsigned int), cudaMemcpyHostToDevice) );
	}

	end_time = gettime();
	printf("cudaMemcpy:\t\t%lf\n",end_time-time);

	time = gettime();
	int iteration = 0;

	printf("Solution is %d\n", solution);

	/* Initialize working set */
	switch ( solution ){
		case 0: case 2: 
			break;
		case 1: case 3:  	
			cudaCheckError( __LINE__, cudaMemset(d_qCounter, 0, sizeof(unsigned int)));
			order_generateQueue_kernel<<<dimGrid, dimBlock>>>(	d_costArray, d_workQueue, d_commit, d_qCounter, 
																qMaxLength, nodeNumber, d_minValue);
			break;
		case 4: case 6:
			generateBitmap_kernel<<<dimGrid, dimBlock>>>(d_frontier, d_update, nodeNumber);
			break;
		case 5: case 7: case 8:
			unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, nodeNumber, d_workQueue, 
																d_qLength, qMaxLength);
			cudaCheckError( __LINE__, cudaMemcpy(&qLength,d_qLength,sizeof(unsigned int), cudaMemcpyDeviceToHost));		
			break;
		case 9:
			unorder_gen_multiQueue_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_update, nodeNumber, 
																d_workQueue, d_qLength, qMaxLength,
																d_workQueue_1, d_qLength_1, qMaxLength_1);
			break;
		default:
			break;
	}
	do
	{				
		//if (iteration==3) break;
		switch ( solution ){
			case 0:	// order+thread+bitmap
					//printf("Thread+Bitmap\n");
					O_T_B_commit_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_costArray, d_edgeArray, 
																d_weightArray, d_commit, d_update,
																nodeNumber, edgeNumber, d_minValue);
					break;
			case 1:	// order+thread+queue
					//printf("Thread+Queue\n");
					cudaCheckError( __LINE__, cudaMemset(d_qCounter, 0, sizeof(unsigned int)));
					order_generateQueue_kernel<<<dimGrid, dimBlock>>>(	d_costArray, d_workQueue, d_commit, d_qCounter, 
																		qMaxLength, nodeNumber, d_minValue);
					O_T_Q_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_costArray, d_edgeArray, d_weightArray,
														 d_update, d_workQueue, d_qCounter, qMaxLength);
					break;
			case 2:	// order+block+bitmap
					//printf("Block+Bitmap\n");
					O_B_B_commit_kernel<<<dimBGrid, dimBBlock>>>(d_vertexArray, d_costArray, d_edgeArray, d_weightArray,
																d_commit, d_update, nodeNumber, edgeNumber, d_minValue);
					break;
			case 3:	// order+block+queue
					//printf("Block+Queue\n");
					cudaCheckError( __LINE__, cudaMemset(d_qCounter, 0, sizeof(unsigned int)));
					order_generateQueue_kernel<<<dimGrid, dimBlock>>>(	d_costArray, d_workQueue, d_commit, d_qCounter, 
																		qMaxLength, nodeNumber, d_minValue);
					O_B_Q_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_costArray, d_edgeArray, d_weightArray,
														d_update, d_workQueue, d_qCounter, qMaxLength);		
					break;
			case 4:	// unorder+thread+bitmap
					//printf("Thread+Bitmap\n");
					unorder_threadBitmap_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_costArray, d_edgeArray, 
																		d_weightArray, d_frontier, d_update, nodeNumber);
					generateBitmap_kernel<<<dimGrid, dimBlock>>>(d_frontier, d_update, nodeNumber);
					break;
			case 5:	// unorder+thread+queue
					//printf("Thread+Queue\n");
					/* Dynamic kernel configuration */
					if (qLength<=maxDegreeT){
						dimGridT.x = 1;
					}
					else if (qLength<=maxDegreeT*MAXDIMGRID){
						dimGridT.x = qLength/maxDegreeT+1;
					}
					else{
						printf("Too many elements in queue\n");
						exit(0);	
					}		
					unorder_threadQueue_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, nodeNumber,
																		d_workQueue, d_qLength);
					cudaCheckError( __LINE__, cudaMemset(d_qLength, 0, sizeof(unsigned int)));
					unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, nodeNumber, d_workQueue, 
																		d_qLength, qMaxLength);
					break;
			case 6:	// unorder+block+bitmap
					//printf("Block+Bitmap\n");
					unorder_blockBitmap_kernel<<<dimBGrid, dimBBlock>>>(d_vertexArray, d_costArray, d_edgeArray, 
																d_weightArray, d_frontier, d_update, nodeNumber);
					cudaCheckError( __LINE__, cudaMemset(d_frontier, 0, sizeof(char)*nodeNumber) );
					generateBitmap_kernel<<<dimGrid, dimBlock>>>(d_frontier, d_update, nodeNumber);
					break;
			case 7:	// unorder+block+queue
					//printf("Block+Queue\n");
					/* Dynamic kernel configuration */
					if (qLength<=MAXDIMGRID){
						dimGridB.x = qLength;
					}
					else if (qLength<=MAXDIMGRID*1024){
						dimGridB.x = MAXDIMGRID;
						dimGridB.y = qLength/MAXDIMGRID+1;
					}
					else{
						printf("Too many elements in queue\n");
						exit(0);	
					}			
					unorder_blockQueue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, nodeNumber,
																		d_workQueue, d_qLength);
					cudaCheckError( __LINE__, cudaMemset(d_qLength, 0, sizeof(unsigned int)));
					unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, nodeNumber, d_workQueue,
																		d_qLength, qMaxLength);
					break;
			case 8: // unordered + thread mapping + queue + delayed buffer
					//printf("Thread+Queue\n");
					/* Dynamic kernel configuration */
					if (qLength<=maxDegreeT){
						dimGridT.x = 1;
					}
					else if (qLength<=maxDegreeT*MAXDIMGRID){
						dimGridT.x = qLength/maxDegreeT+1;
					}
					else{
						printf("Too many elements in queue\n");
						exit(0);	
					}		
					unorder_threadQueue_lb_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, nodeNumber,
																		d_workQueue, d_qLength);
					cudaCheckError( __LINE__, cudaMemset(d_qLength, 0, sizeof(unsigned int)));
					unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, nodeNumber, d_workQueue, 
																		d_qLength, qMaxLength);
					break;

			case 9: // unordered + thread mapping + priority queue
					//printf("Thread+Queue\n");
					/* Dynamic kernel configuration for thread mapping */
					if (qLength<=maxDegreeT){
						dimGridT.x = 1;
					}
					else if (qLength<=maxDegreeT*MAXDIMGRID){
						dimGridT.x = qLength/maxDegreeT+1;
					}
					else{
						printf("Too many elements in queue\n");
						exit(0);
					}
					unorder_threadQueue_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, nodeNumber,
																		d_workQueue, d_qLength);
					/* Dynamic kernel configuration for thread mapping */
					if (qLength_1<=MAXDIMGRID){
						dimGridB.x = qLength_1;
					}
					else if (qLength_1<=MAXDIMGRID*1024){
						dimGridB.x = MAXDIMGRID;
						dimGridB.y = qLength_1/MAXDIMGRID+1;
					}
					else{
						printf("Too many elements in queue\n");
						exit(0);
					}
					unorder_blockQueue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, nodeNumber,
																		d_workQueue_1, d_qLength_1);

					cudaCheckError( __LINE__, cudaMemset(d_qLength, 0, sizeof(unsigned int)));
					cudaCheckError( __LINE__, cudaMemset(d_qLength_1, 0, sizeof(unsigned int)));
					unorder_gen_multiQueue_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_update, nodeNumber, 
																d_workQueue, d_qLength, qMaxLength,
																d_workQueue_1, d_qLength_1, qMaxLength_1);
					break;
			case 10:// unorder+thread queue+dynamic parallelism
					//printf("Thread+Queue\n");
					/* Dynamic kernel configuration */
					if (qLength<=maxDegreeT){
						dimGridT.x = 1;
					}
					else if (qLength<=maxDegreeT*MAXDIMGRID){
						dimGridT.x = qLength/maxDegreeT+1;
					}
					else{
						printf("Too many elements in queue\n");
						exit(0);	
					}		
					unorder_threadQueue_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_costArray, 
																		d_weightArray, d_update, nodeNumber,
																		d_workQueue, d_qLength);
					cudaCheckError( __LINE__, cudaMemset(d_qLength, 0, sizeof(unsigned int)));
					unorder_generateQueue_kernel<<<dimGrid, dimBlock>>>(d_update, nodeNumber, d_workQueue, 
																		d_qLength, qMaxLength);
					break;
			default:
					break;
		}
		if ( solution<4 ){	// order
			findMin(d_costArray, d_commit, d_bufferBlock_1024, d_bufferBlock_1024_1024, d_minValue, nodeNumber);
			cudaCheckError( __LINE__, cudaMemcpy( &minValue, d_minValue, sizeof(int), cudaMemcpyDeviceToHost ) );
			//printf("min value is :%d\n", minValue);
			if ( minValue>=INF )
				break;
		}
		else { // unorder
			if ( solution==4 || solution==6 ){	// bitmap
				cudaCheckError( __LINE__, cudaMemset(d_qLength, 0, sizeof(unsigned int)) );
				countWorkingset_kernel<<<dimGrid, dimBlock>>>(d_update, d_qLength, qMaxLength, nodeNumber);
				cudaCheckError( __LINE__, cudaMemset(d_update, 0, sizeof(char)*nodeNumber) );
			}
			cudaCheckError( __LINE__, cudaMemcpy(&qLength,d_qLength,sizeof(unsigned int), cudaMemcpyDeviceToHost));
			if ( solution==9 ) {
				cudaCheckError( __LINE__, cudaMemcpy(&qLength_1,d_qLength_1,sizeof(unsigned int), cudaMemcpyDeviceToHost));
				qLength = qLength + qLength_1;
			}
			//printf("Working set size is %d\n", qLength);
			if (qLength==0)	break;
		}
		iteration++;		
	}while(1);

	cudaCheckError( __LINE__, cudaMemcpy( costArray, d_costArray, sizeof(int)*nodeNumber, cudaMemcpyDeviceToHost) );
	end_time = gettime();
	printf("SSSP iteration:\t\t%lf\n",end_time-time);

	printf("SSSP terminated in %d iterations\n", iteration);
	cudaFree(d_vertexArray);
	cudaFree(d_costArray);
	cudaFree(d_edgeArray);
	cudaFree(d_weightArray);
}
