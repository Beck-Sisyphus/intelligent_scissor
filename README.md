# intelligent_scissor
computer vision introduction practice on image segmentation

Feb. 28rd, 2018

PAN, Jie && PANG, Sui

## Build
```
mkdir build
cd build
cmake ..
make
./intelligent_scissor
```

Supported Environment:
    macOS High Sierra
    Ubuntu 16.04 LTS
    
Toolchain:
    editor:     CLion
    language:   C++
    compiler:   CMake/make 
    dependency: OpenCV 3.2
    GUI:        Qt

## Core Algorithm
1. Convert the imported picture to cost diagram from its derivative.
2. Store the pixels as nodes, with the following field:
```c++
class Pixel_Node : public FibHeapNode
{
public:
    int state;
    int row, col;

    int link_cost[9];
    long total_cost;

    Pixel_Node* prevNode; // connecting to multiple other nodes called graph
};
```
3. A priority queue implemented by fibonacci heap to store and process the nodes
4. A Dijkstra's algorithm for calculating the minimum path 

## User Interface
We also needs to handle the user interface well for finishing this project. There are three criterias:
1. Handling different type of images, jpeg, png, gif, etc.
2. Handling mouse and keyboard command, as listed here:

   - Ctrl+"+", zoom in;

   - Ctrl+"-", zoom out;

   - Ctrl+Left click first seed;

   - Left click, following seeds;

   - Enter, finish the current contour;

   - Ctrl+Enter, finish the current contour as closed;

   - Backspace, when scissoring, delete the last seed; otherwise, delete selected contour.

3. Store and go in the debug mode.


