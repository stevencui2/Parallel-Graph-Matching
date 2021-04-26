#include <stdio.h>
#include <stdlib.h>

__global__ void exclusive_prefix_sum_gpu(int * oldSum, int * newSum, int distance, int numElements) {
		// Get Thread ID
		const int NUM_THREADS = blockDim.x * gridDim.x;
		const int COL = blockIdx.x * blockDim.x + threadIdx.x;
		const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
		const int FIRST_T_ID = COL + ROW * NUM_THREADS;
	
		for(int curTID = FIRST_T_ID; curTID <= numElements; curTID += NUM_THREADS) {
			if(distance == 0) {
				newSum[curTID] = oldSum[curTID+1];
			} else {
				const int COMPARE_T_ID = curTID+distance > numElements ? numElements : curTID+distance;
				newSum[curTID] = oldSum[curTID] + oldSum[COMPARE_T_ID];
			}
		}
}
