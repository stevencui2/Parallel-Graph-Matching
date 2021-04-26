#include <stdio.h>
#include <stdlib.h>

__global__ void strongestNeighborScan_gpu(
        int * src,
        int * oldDst, int * newDst,
        int * oldWeight, int * newWeight,
        int * madeChanges,
        int distance,
        int numEdges
        ) {
            
    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int FIRST_T_ID = COL + ROW * NUM_THREADS;

    for(int curTID = FIRST_T_ID; curTID <= numEdges; curTID += NUM_THREADS) {
        // get compare thread index, enforce 0 bound
        const int COMPARE_T_ID = curTID - distance > 0 ? curTID - distance : 0;

        // case : shared segment
        if( src[COMPARE_T_ID] == src[curTID]) {
            int strongerIndex;
            const int COMPARE_T_WEIGHT = oldWeight[COMPARE_T_ID];
            const int CUR_T_WEIGHT = oldWeight[curTID];

            if(COMPARE_T_WEIGHT > CUR_T_WEIGHT) {
                strongerIndex = COMPARE_T_ID;
            }
            else if(COMPARE_T_WEIGHT < CUR_T_WEIGHT) {
                strongerIndex = curTID;
            }
            // case: equal weights, take node with smaller vID
            else {
                const int COMPARE_T_D = oldDst[COMPARE_T_ID];
                const int CUR_T_D = oldDst[curTID];

                if(COMPARE_T_D < CUR_T_D) {
                    strongerIndex = COMPARE_T_ID;
                } else {
                    strongerIndex = curTID;
                };
            }

            //Set new destination
            newDst[curTID] = oldDst[strongerIndex];

            //Set new weight
            newWeight[curTID] = oldWeight[strongerIndex];

            if(newDst[curTID] != oldDst[curTID]) { *madeChanges = 1; };
        }
        // case : different segment
        else {
            // defaults to no change
            newDst[curTID] = oldDst[curTID];
            newWeight[curTID] = oldWeight[curTID];
        }
    }
}
