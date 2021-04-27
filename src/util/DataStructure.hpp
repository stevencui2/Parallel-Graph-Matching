/**
 * @file dataStrucutre.hpp
 * @date Spring 2020
 * @author Prof. Zheng Zhang, Teaching Assistants
*/
#ifndef DATASTRUCTURE_H
#define DATASTRUCTURE_H

typedef struct GraphData
{
    int numNodes;
    int numEdges;
    int *src;
    int *dst;
    int *weight;
} GraphData;

#endif
