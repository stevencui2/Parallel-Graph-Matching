/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

int getStrongerThreadIndex(int * oldWeight, int * oldDst, int curTID, int compareTID) {
    int strongerIndex;
    const int COMPARE_T_WEIGHT = oldWeight[compareTID];
    const int CUR_T_WEIGHT = oldWeight[curTID];

    if(COMPARE_T_WEIGHT > CUR_T_WEIGHT) {
        strongerIndex = compareTID;
    }
    else if(COMPARE_T_WEIGHT < CUR_T_WEIGHT) {
        strongerIndex = curTID;
    }
    // case: equal weights, take node with smaller vID
    else {
        const int COMPARE_T_D = oldDst[compareTID];
        const int CUR_T_D = oldDst[curTID];

        if(COMPARE_T_D < CUR_T_D) {
            strongerIndex = compareTID;
        } else {
            strongerIndex = curTID;
        };
    }

    return strongerIndex;
}

__global__ void strongestNeighborScan_gpu(
        int * src,
        int * oldDst, int * newDst,
        int * oldWeight, int * newWeight,
        int * madeChanges,
        int distance,
        int numEdges
        ) {
    // Calculate thread work
    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int THREAD_WORK = numEdges / NUM_THREADS;

    //Get thread ID
    const int FIRST_T_ID = blockIdx.x * blockDim.x + threadIdx.x;

    for(int curTID = FIRST_T_ID; curTID <= numEdges; curTID += NUM_THREADS) {
        // get compare thread index, enforce 0 bound
        const int COMPARE_T_ID = curTID - distance > 0 ? curTID - distance : 0;

        // case : shared segment
        if( src[COMPARE_T_ID] == src[curTID]) {
            const int STRONGER_INDEX = getStrongerThreadIndex(oldWeight, oldDst, curTID, COMPARE_T_ID);
            //Set new destination
            newDst[curTID] = oldDst[STRONGER_INDEX];

            //Set new weight
            newWeight[curTID] = oldWeight[STRONGER_INDEX];

            //Flag any changes
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
