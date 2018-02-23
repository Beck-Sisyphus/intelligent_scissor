//
// Created by beck on 22/2/2018.
//

#ifndef INTELLIGENT_SCISSOR_H
#define INTELLIGENT_SCISSOR_H

#define INF_COST 65535

enum
{
    NODE_INITIAL=1,
    NODE_ACTIVE =2,
    NODE_EXPANDED=3
};

/**
 * Pixel node structure
 * order of the cost changed, different from the counter clockwise in image gradient
 * upper left link as link_cost[0], increment left to right
 * link cost to itself as link_cost[4] is zero
 */
typedef struct Pixel_Node
{
    int state;
    int row, col;

    int link_cost[9];
    long total_cost;

    Pixel_Node* prevNode; // connecting to multiple other nodes called graph

    // For priority queue "push with priority"
    // If the cost is high, the priority is low
    bool operator < (const struct Pixel_Node &another) const {
        return this->total_cost > another.total_cost;
    }
}Pixel_Node;

std::vector<Pixel_Node> node_vector;
cv::Mat image_src, image_gradient, image_path_tree;

#endif //INTELLIGENT_SCISSOR_H
