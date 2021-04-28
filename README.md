# parallel-graph-matching
In this project, I was asked to implement a component of a parallel graph matching algorithm. The program takes a graph file (in matrix-market format) as input and perform graph matching related operations on the GPU.

## Background
### Graph Matching Problem
Given a graph G = (V, E), while V is the set of vertices (also called nodes) and E ⊂ |V|^2. A matching M in G is a set of pairwise non-adjacent edges such that no two edges share a common vertex. A vertex is matched if it is an endpoint of one of the edges in the matching. A vertex is unmatched if it does not belong to any edge in the matching. In Fig. 1, we show examples of possible matchings for a given graph.

A maximum matching can be defined as a matching where the total weight of the edges in the matching is maximized. In Fig. 1, (c) is a maximum matching, where the total weight of the edges in the matching is 7. Fig. (a) and (b) respectively have the total weight of 3 and 2.

<!-- Figure -->

### Parallel Graph Matching
Most well-known matching algorithms such as blossom algorithm are embarrassingly sequen- tial and hard to parallelize. In this project, we will be adopting handshaking-based algorithm that is amenable to parallelization and can be a good fit for GPUs.

In the handshaking-based algorithm, a vertex v extends a hand to one of its neighbours and the neighbor must be sitting on the maximum-weight edge incident to v. If two vertices shake hands, the edge between these two vertices will be added to the matching. An example is shown in Fig. 2 (b) where node A extends a hand to D since edge(A,D) has the largest weight among all edges incident to node A; Nodes C and F shake hands because they extend a hand to each other.

<!-- Figure -->

It is possible that multiple incident edges of a node have maximum weight. In this project, we let the algorithm pick the neighbor vertex that has the smallest vertex index. For example, in Fig. 2(b), among the maximum-weight neighbors of vertex E, we pick vertex B since it has the smallest index (in alphabetical order) among all E’s edges that have maximum-weight 4.

The handshaking algorithm need to run one or multiple passes. A one-pass handshaking checks all nodes once and only once. At the end of every pass, we remove all matched nodes and check if there is any remaining un-matched nodes. If the remaining un-matched nodes are connected, another pass of handshaking must be performed. We repeat this until no more edges can be added to the matching. In Fig. 2, we show two passes of handshaking.

The handshaking algorithm is highly data parallel, since each vertex is processed indepen- dently and the find maximum-weight-neighbor step involves only reads to shared data but no writes. It is a greedy algorithm that attempts to maximize the total weight in the matching.

