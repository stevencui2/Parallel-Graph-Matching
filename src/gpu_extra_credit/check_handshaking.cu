#include <stdio.h>
#include <stdlib.h>

__global__ void check_handshaking_gpu(int * strongNeighbor, int * matches, int numNodes) {
	// Get Thread ID
	const int NUM_THREADS = blockDim.x * gridDim.x;
	const int COL = blockIdx.x * blockDim.x + threadIdx.x;
	const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	const int FIRST_T_ID = COL + ROW * NUM_THREADS;

	for(int curTID = FIRST_T_ID; curTID <= numNodes; curTID += NUM_THREADS) {
		if(strongNeighbor[curTID] == strongNeighbor[curTID+1] && matches[curTID] == -1){
			matches[curTID] = strongNeighbor[curTID+1];
		}
	}
}
