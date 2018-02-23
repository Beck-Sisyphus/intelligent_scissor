//
// Created by beck on 21/2/2018.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <iostream>
#include <queue>
#include <vector>
#include "scissor.h"

using namespace cv;
using namespace std;

String image_directory = "../image/avatar.jpg";
String cost_graph_directory = "../image/avatar_cost_graph.jpg";


#define DEBUG

//#define COST_GRAPH

/**
 * 20180221 Beck Pang, implementing intensity derivative
 * diagonal link,   D(link1)=| img(i+1,j) - img(i,j-1) |/sqrt(2)
 * horizontal link, D(link0)=|(img(i,j-1) + img(i+1,j-1))/2 - (img(i,j+1) + img(i+1,j+1))/2|/2
 * vertical link,   D(link2)=|(img(i-1,j) + img(i-1,j-1))/2 - (img(i+1,j) + img(i+1,j-1))/2|/2
 * And in the end,  cost(link)=(maxD-D(link))*length(link)
 */
int calculate_cost_image()
{
    int rows, cols; // coordinate of the pixel
    rows = image_src.rows;
    cols = image_src.cols;
#ifdef DEBUG
    cout << "rows = "  << rows << ", cols = " << cols << endl;
#endif

    // a new picture with nine times the size of original picture, all white pixels
    image_gradient = Mat( (rows-2) * 3, (cols-2) * 3, CV_8UC3, Scalar(255, 255, 255));
//    image_gradient.at<Vec3b>( 937,1267 )[0] = 0;

    double D_square[8] = {0};
    int link[8];    // local derivative
    int maxD = 0;   // global maximum derivative
    Vec3b pixel[8];

    int i, j, k, l; // iterators for the original picture
    int x, y;       // iterators for the gradient
    for ( i = 1; i < rows - 1; ++i) {
        for ( j = 1; j < cols - 1; ++j) {
            // initialize
            for ( k = 0; k < 8; ++k) {
                link[k]  = 0;
                D_square[k] = 0;
            }

            //// diagonal link,   D(link1)=| img(i+1,j) - img(i,j-1) |/sqrt(2)
            // x + 1, y - 1
            pixel[0] = image_src.at<Vec3b>( i+1 , j   );
            pixel[1] = image_src.at<Vec3b>( i   , j-1 );

            // x - 1, y - 1
            pixel[2] = image_src.at<Vec3b>( i   , j-1 );
            pixel[3] = image_src.at<Vec3b>( i-1 , j   );

            // x - 1, y + 1
            pixel[4] = image_src.at<Vec3b>( i-1 , j   );
            pixel[5] = image_src.at<Vec3b>( i   , j+1 );

            // x + 1, y + 1
            pixel[6] = image_src.at<Vec3b>( i   , j+1 );
            pixel[7] = image_src.at<Vec3b>( i+1 , j   );

            // Calculate link[1],[3],[5],[7]
            for ( k = 0; k < 4; ++k) {
                int m = 2 * k + 1;
                for ( l = 0; l < 3; ++l) {
                    D_square[m] += pow(pixel[m - 1][l] - pixel[m][l], 2);
                }
                link[m] = (int)sqrt(D_square[m] / 6);
            }

            //// horizontal link, D(link0)=|(img(i,j-1) + img(i+1,j-1))/2 - (img(i,j+1) + img(i+1,j+1))/2|/2
            // x + 1, y
            pixel[0] = image_src.at<Vec3b>( i   , j-1 );
            pixel[1] = image_src.at<Vec3b>( i+1 , j-1 );
            pixel[2] = image_src.at<Vec3b>( i   , j+1 );
            pixel[3] = image_src.at<Vec3b>( i+1 , j+1 );

            // x - 1, y
            pixel[4] = image_src.at<Vec3b>( i   , j-1 );
            pixel[5] = image_src.at<Vec3b>( i-1 , j-1 );
            pixel[6] = image_src.at<Vec3b>( i   , j+1 );
            pixel[7] = image_src.at<Vec3b>( i-1 , j+1 );

            for ( l = 0; l < 3; ++l) {
                D_square[0] += pow( (pixel[0][l] + pixel[1][l])/2 - (pixel[2][l] + pixel[3][l])/2 , 2);
                D_square[4] += pow( (pixel[4][l] + pixel[5][l])/2 - (pixel[6][l] + pixel[7][l])/2 , 2);
            }
            link[0] = (int)sqrt(D_square[0] / 12);
            link[4] = (int)sqrt(D_square[4] / 12);

            //// vertical link,   D(link2)=|(img(i-1,j) + img(i-1,j-1))/2 - (img(i+1,j) + img(i+1,j-1))/2|/2.
            // x    , y - 1
            pixel[0] = image_src.at<Vec3b>( i-1 , j   );
            pixel[1] = image_src.at<Vec3b>( i-1 , j-1 );
            pixel[2] = image_src.at<Vec3b>( i+1 , j   );
            pixel[3] = image_src.at<Vec3b>( i+1 , j-1 );

            // x    , y + 1
            pixel[4] = image_src.at<Vec3b>( i+1 , j   );
            pixel[5] = image_src.at<Vec3b>( i+1 , j+1 );
            pixel[6] = image_src.at<Vec3b>( i-1 , j   );
            pixel[7] = image_src.at<Vec3b>( i-1 , j+1 );


            for ( l = 0; l < 3; ++l) {
                D_square[2] += pow( (pixel[0][l] + pixel[1][l])/2 - (pixel[2][l] + pixel[3][l])/2 , 2);
                D_square[6] += pow( (pixel[4][l] + pixel[5][l])/2 - (pixel[6][l] + pixel[7][l])/2 , 2);
            }
            link[2] = (int)sqrt(D_square[2] / 12);
            link[6] = (int)sqrt(D_square[6] / 12);

            //// Find maxD and add the cost graph
            for ( k = 0; k < 8; ++k) {
                if (link[l] > maxD)
                    maxD = link[l];
            }

            x = i * 3 - 2;
            y = j * 3 - 2;

            for ( k = 0; k < 3; ++k) {
                image_gradient.at<Vec3b>( x  , y   )[k] = 255;
                image_gradient.at<Vec3b>( x+1, y-1 )[k] = (uchar)link[0];
                image_gradient.at<Vec3b>( x-1, y-1 )[k] = (uchar)link[1];
                image_gradient.at<Vec3b>( x-1, y+1 )[k] = (uchar)link[2];
                image_gradient.at<Vec3b>( x+1, y+1 )[k] = (uchar)link[3];
                image_gradient.at<Vec3b>( x+1, y   )[k] = (uchar)link[4];
                image_gradient.at<Vec3b>( x-1, y   )[k] = (uchar)link[5];
                image_gradient.at<Vec3b>( x  , y-1 )[k] = (uchar)link[6];
                image_gradient.at<Vec3b>( x  , y+1 )[k] = (uchar)link[7];
            }
        }
    }


    for ( i = 1; i < rows - 1; ++i) {
        for ( j = 1; j < cols - 1; ++j) {
            x = i * 3 - 2;
            y = j * 3 - 2;
            //// update cost, cost(link)=(maxD - D(link)) * length(link)
            for ( k = 0; k < 3; ++k) {
                image_gradient.at<Vec3b>( x  , y   )[k] =                  image_gradient.at<Vec3b>( x  , y   )[k];
                image_gradient.at<Vec3b>( x+1, y-1 )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x+1, y-1 )[k]) * sqrt(2) );
                image_gradient.at<Vec3b>( x-1, y-1 )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x-1, y-1 )[k]) * sqrt(2) );
                image_gradient.at<Vec3b>( x-1, y+1 )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x-1, y+1 )[k]) * sqrt(2) );
                image_gradient.at<Vec3b>( x+1, y+1 )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x+1, y+1 )[k]) * sqrt(2) );
                image_gradient.at<Vec3b>( x+1, y   )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x+1, y   )[k]) );
                image_gradient.at<Vec3b>( x-1, y   )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x-1, y   )[k]) );
                image_gradient.at<Vec3b>( x  , y-1 )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x  , y-1 )[k]) );
                image_gradient.at<Vec3b>( x  , y+1 )[k] = (uchar)( (maxD - image_gradient.at<Vec3b>( x  , y+1 )[k]) );
            }

#ifdef DEBUG
//            cout << "x = "  << x << ", y =" << y << endl;
//            cout << "  " << +(uchar)link[0] << ", " << +(uchar)link[1] << endl;
//            cout << "  " << +(uchar)link[2] << ", " << +(uchar)link[3] << endl;
//            cout << "  " << +(uchar)link[4] << ", " << +(uchar)link[5] << endl;
//            cout << "  " << +(uchar)link[6] << ", " << +(uchar)link[7] << endl;
#endif
        }
    }

#ifdef DEBUG
     cout << "maxD = " << maxD << endl;
#endif

#ifdef COST_GRAPH
    // Create a window
    namedWindow("gradient window", WINDOW_AUTOSIZE);
    imshow("gradient window", image_gradient);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    try {
        imwrite(cost_graph_directory, image_gradient, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 1;
    }
    fprintf(stdout, "Saved PNG file with alpha data.\n");
#endif
    return 0;
}

void init_node_vector()
{
    node_vector.clear();

    int rows, cols; // coordinate of the pixel
    rows = image_src.rows;
    cols = image_src.cols;

    // preallocate the vector to save memory allocation time
    node_vector.reserve((unsigned long)rows * cols);

    int i, j, k, m;
    int x, y;
    for ( i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            Pixel_Node pixel_node = {};
            pixel_node.state        = NODE_INITIAL;
            pixel_node.row          = i;
            pixel_node.col          = j;
            pixel_node.total_cost   = INF_COST;
            pixel_node.prevNode     = nullptr;

            // Set link cost for normal and edge case
            if ( i == 0 || i == rows-1 || j == 0 || j == cols-1 )
            {
                for (k = 0; k < 9; ++k)
                    pixel_node.link_cost[k] = INF_COST;
            }
            else
            {
                x = i * 3 - 2;
                y = j * 3 - 2;
                int count = 0;
                for (k = -1; k <= 1; ++k) {
                    for (m = -1; m <= 1; ++m) {
                        if  (k == 0 && m == 0)
                            pixel_node.link_cost[count] = 0;
                        else
                            pixel_node.link_cost[count] = image_gradient.at<Vec3b>( x+k, y+m )[0];
                        count++;
                    }
                }
            }

            node_vector.push_back(pixel_node);
        }
    }
}

priority_queue<Pixel_Node> active_nodes = {};

/**
 * minimum cost path
 * a recursive function from dijkstra's algorithm
 * @input seed
 * @input graph
 * @output return_dist
 * @output return_prev
 */
bool minimum_cost_path_dijkstra(Point* seed, Point* dest, void* return_dist, void* return_prev)
{
    int rows, cols; // coordinate of the pixel
    rows = image_src.rows;
    cols = image_src.cols;

    auto seed_source = seed->x * rows + seed->y;
    Pixel_Node root = node_vector[seed_source];

    root.total_cost = 0;

    active_nodes.push(root);

#ifdef DEBUG
//    for (int i = 0; i < rows - 1; ++i) {
//        for (int j = 0; j < cols - 1; ++j) {
//            active_nodes.push(node_vector[i * rows + j]);
//        }
//    }
//
//    Pixel_Node current = active_nodes.top();
//    active_nodes.pop(); // remove the top element
//    cout << "cost for first on the queue = "<< current.total_cost << endl;
//
//    Pixel_Node second = active_nodes.top();
//    active_nodes.pop(); // remove the top element
//    cout << "cost for first on the queue = "<< second.total_cost << endl;
#endif

    while ( !active_nodes.empty() )
    {
        Pixel_Node current = active_nodes.top();
        active_nodes.pop(); // remove the top element

        current.state = NODE_EXPANDED;

        if (current.row == dest->x && current.col == dest->y)
        {
            // reached destination
            return true;
        }

        int i, j;
        int index;
        // Expand its neighbor nodes
        for ( i = 0; i < 3; ++i) {
            for ( j = 0; j < 3; ++j) {
                index = (current.row + i - 1) * rows + (current.col + j - 1);
                Pixel_Node* neighbor = &node_vector[index];
                if (neighbor->state == NODE_INITIAL)
                {
                    neighbor->prevNode   = &current;
                    neighbor->total_cost = current.total_cost + current.link_cost[i * 3 + j];
                    neighbor->state      = NODE_ACTIVE;
                    active_nodes.push(*neighbor);
                }
                else if (neighbor->state == NODE_ACTIVE)
                {
                    if (current.total_cost + current.link_cost[i * 3 + j] < neighbor->total_cost)
                    {
                        neighbor->prevNode   = &current;
                        neighbor->total_cost = current.total_cost + current.link_cost[i * 3 + j];
                        // TODO: original node never gets updated
                        active_nodes.push(*neighbor);
                    }
                }
            }
        }
    }
    return false;
}

/*
 * OpenCV UI part, handling mouse actions
 */
void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
    cout << "mouse move over the window at position ("<< x <<", "<< y <<")" << endl;
    switch (event) {
        case EVENT_MOUSEMOVE    : // Track the mouse position x, y
            break;
        case EVENT_LBUTTONDOWN  : // place a seed
            break;
        case EVENT_RBUTTONDOWN  :  break;
        case EVENT_MBUTTONDOWN  :  break;
        case EVENT_LBUTTONUP    :  break;
        case EVENT_RBUTTONUP    :  break;
        case EVENT_MBUTTONUP    :  break;
        case EVENT_LBUTTONDBLCLK:  // reset the contour
            break;
        case EVENT_RBUTTONDBLCLK:  break;
        case EVENT_MBUTTONDBLCLK:  break;
        case EVENT_MOUSEWHEEL   : // zoom in
            break;
        case EVENT_MOUSEHWHEEL  : // zoom out
            break;
        default: break;
    }
}

int main( int argc, char** argv )
{
    image_src = imread(image_directory, CV_LOAD_IMAGE_COLOR);
//    image_shown = image_src.clone();

    if (! image_src.data)
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Create a window
    namedWindow("Display window", WINDOW_AUTOSIZE);

    // set the callback function for any mouse event
    setMouseCallback("Display window", mouse_callback, nullptr);

    // show the image
    imshow("Display window", image_src);

    //// Algorithm part
    calculate_cost_image();

    init_node_vector();

    Point seed_point(200, 200);
    Point dest_point(300, 300);
    int dist;
    int prev;
    bool result = minimum_cost_path_dijkstra(&seed_point, &dest_point,(void*) &dist, (void*) &prev);
    cout << "dijkstra result " << result << endl;

    waitKey(10000);
    return 0;
}