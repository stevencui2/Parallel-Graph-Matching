/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void collateSegments_gpu(int * src, int * scanResult, int * output, int numEdges) {

    // Get Thread ID
    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int FIRST_T_ID = COL + ROW * NUM_THREADS;

    //Terminate if thread ID is larger than array
    if(FIRST_T_ID >= numEdges) return;

    for(int curTID = FIRST_T_ID; curTID <= numEdges; curTID += NUM_THREADS) {
        if(src[curTID] != src[curTID+1]) {
            output[src[curTID]] = scanResult[curTID];
        }
    }
}
