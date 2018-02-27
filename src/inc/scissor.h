//
// Created by beck on 22/2/2018.
//

#ifndef INTELLIGENT_SCISSOR_H
#define INTELLIGENT_SCISSOR_H

#include "pixel_node.h"

int calculate_cost_image();
void init_node_vector();
bool minimum_cost_path_dijkstra(cv::Point* seed, vector<Pixel_Node*> *nodes_graph);

// Global node graph
std::vector<Pixel_Node*> node_vector;
cv::Mat image_src, image_gradient, image_path_tree;

#endif //INTELLIGENT_SCISSOR_H
