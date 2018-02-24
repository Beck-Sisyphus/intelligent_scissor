//
// Created by beck on 22/2/2018.
//

#ifndef INTELLIGENT_SCISSOR_H
#define INTELLIGENT_SCISSOR_H

#include "fibheap.h"

#define INF_COST 0x0FFFFFFF

//enum
//{
//    NODE_INITIAL=1,
//    NODE_ACTIVE =2,
//    NODE_EXPANDED=3
//};

/**
 * Pixel node structure
 * order of the cost changed, different from the counter clockwise in image gradient
 * upper left link as link_cost[0], increment left to right
 * link cost to itself as link_cost[4] is zero
 */
class Pixel_Node : public FibHeapNode
{
public:
    int state;
    int row, col;

    int link_cost[9];
    long total_cost;

    Pixel_Node* prevNode; // connecting to multiple other nodes called graph

    enum Node_state{INITIAL, ACTIVE, EXPANDED};
    // constructor
    Pixel_Node(int row, int col) : FibHeapNode()
    {
        this->state        = INITIAL;
        this->row          = row;
        this->col          = col;
        this->total_cost   = INF_COST;
        this->prevNode     = nullptr;
        for (int i = 0; i < 9; ++i) { this->link_cost[i] = INF_COST; }
    }

    virtual void operator =  (FibHeapNode &RHS);
    virtual int  operator == (FibHeapNode &RHS);
    virtual int  operator <  (FibHeapNode &RHS);

    virtual void Print();
};

void Pixel_Node::Print()
{
    FibHeapNode::Print();
    cout << "state: " << state << " row: " << row << " col: " << col << " cost: " << total_cost << endl;
}

void Pixel_Node::operator = (FibHeapNode &RHS)
{
    auto pRHS = (Pixel_Node&) RHS;
    FHN_Assign(RHS);
    this->state  = pRHS.state;
    this->row    = pRHS.row;
    this->col    = pRHS.col;
    this->total_cost = pRHS.total_cost;
    this->prevNode   = pRHS.prevNode;
    for (int i = 0; i < 9; ++i) { this->link_cost[i] = pRHS.link_cost[i]; }
}

int Pixel_Node::operator == (FibHeapNode &RHS)
{
    auto pRHS = (Pixel_Node&) RHS;

    // Make sure both sides are not negative infinite
    if (FHN_Cmp(RHS)) return 0;

//    return (this->row == pRHS.row && this->col == pRHS.col) ? 1 : 0;

    // Misunderstand the ==, should be comparing the cost
    return total_cost == pRHS.total_cost;
}


int Pixel_Node::operator < (FibHeapNode &RHS)
{
    int X;
    if ((X = FHN_Cmp(RHS)) != 0)
        return X < 0 ? 1 : 0;

    // For priority queue "push with priority", if the cost is high, the priority is low
    // For fibonacci node, the priority is high when the return number is large
    return this->total_cost < ((Pixel_Node&) RHS).total_cost ? 1 : 0;
}

std::vector<Pixel_Node*> node_vector;
cv::Mat image_src, image_gradient, image_path_tree;

#endif //INTELLIGENT_SCISSOR_H
