//
// Created by beck on 21/2/2018.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <cstdio>
#include <iostream>
#include <vector>
#include <stack>
#include <chrono>
#include <ctime>

#include "scissor.h"
#include "plot.h"

#define DEBUG

//#define DEBUG_NODE_VECTOR

//#define DEBUG_DIJKSTRA

//#define TEST_DIJKSTRA

//#define COST_GRAPH

//#define PATH_TREE_DEBUG

#define DEBUG_USER_INTERFACE

using namespace cv;
using namespace std;

String image_directory = "../image/avatar.jpg";
extern String plot_window_name;
/**
 * 20180221 Beck Pang, implementing intensity derivative
 * diagonal link,   D(link1)=| img(i+1,j) - img(i,j-1) |/sqrt(2)
 * horizontal link, D(link0)=|(img(i,j-1) + img(i+1,j-1))/2 - (img(i,j+1) + img(i+1,j+1))/2|/2
 * vertical link,   D(link2)=|(img(i-1,j) + img(i-1,j-1))/2 - (img(i+1,j) + img(i+1,j-1))/2|/2
 * And in the end,  cost(link)=(maxD-D(link))*length(link)
 * @input: image source
 * @output: image gradient
 */
int calculate_cost_image(Mat* image_src, Mat* image_gradient)
{
    int rows, cols; // coordinate of the pixel
    rows = image_src->rows;
    cols = image_src->cols;
#ifdef DEBUG
    cout << "rows = " << rows << ", cols = " << cols << endl;
#endif

    // a new picture with nine times the size of original picture, all white pixels
    *image_gradient = Mat((rows - 2) * 3, (cols - 2) * 3, CV_8UC3, Scalar(255, 255, 255));
//    image_gradient->at<Vec3b>( 937,1267 )[0] = 0;

    double D_square[8] = {0};
    int link[8];    // local derivative
    int maxD = 0;   // global maximum derivative
    Vec3b pixel[8];

    int i, j, k, l; // iterators for the original picture
    int x, y;       // iterators for the gradient
    for (i = 1; i < rows - 1; ++i) {
        for (j = 1; j < cols - 1; ++j) {
            // initialize
            for (k = 0; k < 8; ++k) {
                link[k] = 0;
                D_square[k] = 0;
            }

            //// diagonal link,   D(link1)=| img(i+1,j) - img(i,j-1) |/sqrt(2)
            // x + 1, y - 1
            pixel[0] = image_src->at<Vec3b>(i + 1, j);
            pixel[1] = image_src->at<Vec3b>(i, j - 1);

            // x - 1, y - 1
            pixel[2] = image_src->at<Vec3b>(i, j - 1);
            pixel[3] = image_src->at<Vec3b>(i - 1, j);

            // x - 1, y + 1
            pixel[4] = image_src->at<Vec3b>(i - 1, j);
            pixel[5] = image_src->at<Vec3b>(i, j + 1);

            // x + 1, y + 1
            pixel[6] = image_src->at<Vec3b>(i, j + 1);
            pixel[7] = image_src->at<Vec3b>(i + 1, j);

            // Calculate link[1],[3],[5],[7]
            for (k = 0; k < 4; ++k) {
                int m = 2 * k + 1;
                for (l = 0; l < 3; ++l) {
                    D_square[m] += pow(pixel[m - 1][l] - pixel[m][l], 2);
                }
                link[m] = (int) sqrt(D_square[m] / 6);
            }

            //// horizontal link, D(link0)=|(img(i,j-1) + img(i+1,j-1))/2 - (img(i,j+1) + img(i+1,j+1))/2|/2
            // x + 1, y
            pixel[0] = image_src->at<Vec3b>(i, j - 1);
            pixel[1] = image_src->at<Vec3b>(i + 1, j - 1);
            pixel[2] = image_src->at<Vec3b>(i, j + 1);
            pixel[3] = image_src->at<Vec3b>(i + 1, j + 1);

            // x - 1, y
            pixel[4] = image_src->at<Vec3b>(i, j - 1);
            pixel[5] = image_src->at<Vec3b>(i - 1, j - 1);
            pixel[6] = image_src->at<Vec3b>(i, j + 1);
            pixel[7] = image_src->at<Vec3b>(i - 1, j + 1);

            for (l = 0; l < 3; ++l) {
                D_square[0] += pow((pixel[0][l] + pixel[1][l]) / 2 - (pixel[2][l] + pixel[3][l]) / 2, 2);
                D_square[4] += pow((pixel[4][l] + pixel[5][l]) / 2 - (pixel[6][l] + pixel[7][l]) / 2, 2);
            }
            link[0] = (int) sqrt(D_square[0] / 12);
            link[4] = (int) sqrt(D_square[4] / 12);

            //// vertical link,   D(link2)=|(img(i-1,j) + img(i-1,j-1))/2 - (img(i+1,j) + img(i+1,j-1))/2|/2.
            // x    , y - 1
            pixel[0] = image_src->at<Vec3b>(i - 1, j);
            pixel[1] = image_src->at<Vec3b>(i - 1, j - 1);
            pixel[2] = image_src->at<Vec3b>(i + 1, j);
            pixel[3] = image_src->at<Vec3b>(i + 1, j - 1);

            // x    , y + 1
            pixel[4] = image_src->at<Vec3b>(i + 1, j);
            pixel[5] = image_src->at<Vec3b>(i + 1, j + 1);
            pixel[6] = image_src->at<Vec3b>(i - 1, j);
            pixel[7] = image_src->at<Vec3b>(i - 1, j + 1);


            for (l = 0; l < 3; ++l) {
                D_square[2] += pow((pixel[0][l] + pixel[1][l]) / 2 - (pixel[2][l] + pixel[3][l]) / 2, 2);
                D_square[6] += pow((pixel[4][l] + pixel[5][l]) / 2 - (pixel[6][l] + pixel[7][l]) / 2, 2);
            }
            link[2] = (int) sqrt(D_square[2] / 12);
            link[6] = (int) sqrt(D_square[6] / 12);

            //// Find maxD and add the cost graph
            for (k = 0; k < 8; ++k) {
                if (link[l] > maxD)
                    maxD = link[l];
            }

            x = i * 3 - 2;
            y = j * 3 - 2;

            for (k = 0; k < 3; ++k) {
                image_gradient->at<Vec3b>(x, y)[k] = 255;
                image_gradient->at<Vec3b>(x + 1, y - 1)[k] = (uchar) link[0];
                image_gradient->at<Vec3b>(x - 1, y - 1)[k] = (uchar) link[1];
                image_gradient->at<Vec3b>(x - 1, y + 1)[k] = (uchar) link[2];
                image_gradient->at<Vec3b>(x + 1, y + 1)[k] = (uchar) link[3];
                image_gradient->at<Vec3b>(x + 1, y)[k] = (uchar) link[4];
                image_gradient->at<Vec3b>(x - 1, y)[k] = (uchar) link[5];
                image_gradient->at<Vec3b>(x, y - 1)[k] = (uchar) link[6];
                image_gradient->at<Vec3b>(x, y + 1)[k] = (uchar) link[7];
            }
        }
    }


    for (i = 1; i < rows - 1; ++i) {
        for (j = 1; j < cols - 1; ++j) {
            x = i * 3 - 2;
            y = j * 3 - 2;
            //// update cost, cost(link)=(maxD - D(link)) * length(link)
            for (k = 0; k < 3; ++k) {
                image_gradient->at<Vec3b>(x, y)[k] = image_gradient->at<Vec3b>(x, y)[k];
                image_gradient->at<Vec3b>(x + 1, y - 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x + 1, y - 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x - 1, y - 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x - 1, y - 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x - 1, y + 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x - 1, y + 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x + 1, y + 1)[k] = (uchar) (
                        (maxD - image_gradient->at<Vec3b>(x + 1, y + 1)[k]) * sqrt(2));
                image_gradient->at<Vec3b>(x + 1, y)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x + 1, y)[k]));
                image_gradient->at<Vec3b>(x - 1, y)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x - 1, y)[k]));
                image_gradient->at<Vec3b>(x, y - 1)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x, y - 1)[k]));
                image_gradient->at<Vec3b>(x, y + 1)[k] = (uchar) ((maxD - image_gradient->at<Vec3b>(x, y + 1)[k]));
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

    return 0;
}

/**
 * initialize all linked cost for the complete image from image gradient
 * @output node_vector
 * @input image_gradient
 */
void init_node_vector(int rows, int cols, vector<Pixel_Node *> *node_vector, Mat* image_gradient)
{
    node_vector->clear();

    // preallocate the vector to save memory allocation time
    node_vector->reserve((unsigned long) rows * cols);

    int i, j, k, m;
    int x, y;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            auto pixel_node = new Pixel_Node(i, j);

            // Set link cost for normal and edge case
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                for (k = 0; k < 9; ++k)
                    pixel_node->link_cost[k] = INF_COST;
            } else {
                x = i * 3 - 2;
                y = j * 3 - 2;
                int count = 0;
                for (k = -1; k <= 1; ++k) {
                    for (m = -1; m <= 1; ++m) {
                        if (k == 0 && m == 0)
                            pixel_node->link_cost[count] = INF_COST;
                        else
                            pixel_node->link_cost[count] = image_gradient->at<Vec3b>(x + k, y + m)[0];
                        count++;
                    }
                }
            }

            node_vector->push_back(pixel_node);
        }
    }
#ifdef DEBUG_NODE_VECTOR
    int expected_x, expected_y;
    for ( expected_x = 0; expected_x < rows/10; ++expected_x) {
        for ( expected_y = 0; expected_y < cols/10; ++expected_y) {
            auto seed_source = expected_x * cols + expected_y;
            Pixel_Node* current = node_vector[seed_source];
            cout << "expected_x = " << expected_x << " expected_y = " << expected_y << endl;
            current->Print();
        }
    }
#endif
}

/**
 * @brief calculate a minimum cost path for the seed point within a picture
 *          a recursive function from dijkstra's algorithm
 * @input seed, pixel coordinate for a picture
 * @output nodes_graph
 */
bool minimum_cost_path_dijkstra(int rows, int cols, Point *seed, vector<Pixel_Node *> *nodes_graph)
{
    FibHeap active_nodes; // Local priority heap that will be empty in the end

    auto seed_source = seed->x * cols + seed->y;
    Pixel_Node *root = nodes_graph->data()[seed_source];
    root->total_cost = 0;
    active_nodes.Insert(root);

    while (active_nodes.GetNumNodes() > 0) {
        auto current = (Pixel_Node *) active_nodes.ExtractMin();

//        cout << "number of nodes: " << active_nodes.GetNumNodes() << endl;
//        current->Print();

        current->state = Pixel_Node::EXPANDED;

//        if (current->row == dest->x && current->col == dest->y)
//        {
//            // reached destination
//            return true;
//        }

        int i, j;
        int index;
        int x_now, y_now;
        // Expand its neighbor nodes
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                x_now = current->row + i - 1;
                y_now = current->col + j - 1;

                // Keep the index within boundary
                if (x_now >= 0 && x_now < rows && y_now >= 0 && y_now < cols) {
                    index = x_now * cols + y_now;
                    Pixel_Node *neighbor = nodes_graph->data()[index];

//                    neighbor->Print();

                    if (neighbor->state == Pixel_Node::INITIAL) {
                        neighbor->prevNode = current;
                        neighbor->total_cost = current->total_cost + current->link_cost[i * 3 + j];
                        neighbor->state = Pixel_Node::ACTIVE;
                        active_nodes.Insert(neighbor);
                    } else if (neighbor->state == Pixel_Node::ACTIVE) {
                        if (current->total_cost + current->link_cost[i * 3 + j] < neighbor->total_cost) {
                            Pixel_Node new_node(neighbor->row, neighbor->col);
                            new_node = *neighbor; // Get a copy of the original node
                            new_node.total_cost = current->total_cost + current->link_cost[i * 3 + j];
                            new_node.prevNode = current;
                            active_nodes.DecreaseKey(neighbor, new_node);
                        }
                    }
                }
            }
        }
    }
    return true;
}

stack< Point > points_stack;
stack< Mat >   images_stack;
stack< vector<Pixel_Node*> > graphs_stack;
extern Scalar point_to_point_color;
extern Scalar point_to_path_color;

int click_count = 0;

/**
 * OpenCV UI part, handling mouse actions
 */
static void mouse_callback(int event, int x, int y, int flags, void *userdata)
{
    auto coordinate = (int*) userdata;
    int rows = coordinate[0];
    int cols = coordinate[1];
    bool clicked = false;
    bool mouse_moved   = false;

    switch (event) {
        case EVENT_MOUSEMOVE    :
            mouse_moved = true;
            break;
        case EVENT_LBUTTONDOWN  :
            clicked = true;
            break;
        case EVENT_RBUTTONDOWN  :
            break;
        case EVENT_MBUTTONDOWN  :
            break;
        case EVENT_LBUTTONUP    :
            break;
        case EVENT_RBUTTONUP    :
            break;
        case EVENT_MBUTTONUP    :
            break;
        case EVENT_LBUTTONDBLCLK:  // reset the contour
            break;
        case EVENT_RBUTTONDBLCLK:
            break;
        case EVENT_MBUTTONDBLCLK:
            break;
        case EVENT_MOUSEWHEEL   : // zoom in
            break;
        case EVENT_MOUSEHWHEEL  : // zoom out
            break;
        default:
            break;
    }

    // Update the image in every mouse event
    // NOTE: point x is col number, y is row number
    auto mouse_point = Point(y, x);
    Mat current_image = images_stack.top().clone();

#ifdef DEBUG_USER_INTERFACE
    auto start = std::chrono::system_clock::now();
    cout << " x: " << x << " y: " << y << endl;
#endif

    // check the edge to safely draw the circle
    if (x > 1 && x < coordinate[1]-2 && y > 1 && y < coordinate[0]-2) {
        Point seed_flip = Point(mouse_point.y, mouse_point.x);
        circle(current_image, seed_flip, 2, point_to_point_color, 2);

        if (mouse_moved)
        {
            if ( !points_stack.empty() ) {
                Point* seed;
                vector<Pixel_Node *>* seed_graph;
                seed = &points_stack.top();
                seed_graph = &graphs_stack.top();
                assert(seed != nullptr);
                assert(seed_graph != nullptr);

                plot_path_tree_point_to_point(seed, &mouse_point, seed_graph, &current_image);
            }
        }

        if (clicked)
        {

            vector<Pixel_Node *> nodes_graph;
            init_node_vector(rows, cols, &nodes_graph, &image_gradient);

            minimum_cost_path_dijkstra(rows, cols, &mouse_point, &nodes_graph);

            if ( !points_stack.empty() ) {
                Point stack = points_stack.top();
                cout << "clicked: point on stack: " << stack.x << " " << stack.y << " mouse point " << mouse_point.x << " " << mouse_point.y << endl;

                // Draw from the past saved node to the current clicked node
                plot_path_tree_point_to_point(&points_stack.top(), &mouse_point, &graphs_stack.top(), &current_image);
            }

            // Update the stacks
            points_stack.push(mouse_point);
            images_stack.push(current_image);
            graphs_stack.push(nodes_graph);

#ifdef DEBUG_USER_INTERFACE
            auto end = std::chrono::system_clock::now();
            cout << "points_stack size " << points_stack.size() << endl;
            std::chrono::duration<double> running_seconds = end - start;
            cout << "dijkstra result " << ++click_count << " with " << running_seconds.count() << "seconds. " << endl;
#endif
        }
    }

    imshow(plot_window_name, current_image);
}

int main(int argc, char **argv)
{
    image_src  = imread(image_directory, CV_LOAD_IMAGE_COLOR);

    if (!image_src.data) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    images_stack.push(image_src);

    int rows, cols; // coordinate of the pixel
    rows = image_src.rows;
    cols = image_src.cols;
    int coordinate[2];
    coordinate[0] = image_src.rows;
    coordinate[1] = image_src.cols;

    // Create the point to point path
    namedWindow(plot_window_name, WINDOW_AUTOSIZE);

    imshow(plot_window_name, images_stack.top());

    // set the callback function for any mouse event
    setMouseCallback(plot_window_name, mouse_callback, coordinate);

    //// Algorithm part
    calculate_cost_image(&image_src, &image_gradient);

//    init_node_vector(rows, cols, &node_vector_original, &image_gradient);

    // Pre allocate the space to cut run time
    Point seed_point = Point(100, 100);
    vector<Pixel_Node *> space_malloc_graph;
    init_node_vector(rows, cols, &space_malloc_graph, &image_gradient);
    minimum_cost_path_dijkstra(rows, cols, &seed_point, &space_malloc_graph);

    cout << "Init the seed point graph " << graphs_stack.size() << endl;

#ifdef COST_GRAPH
    // Create a window
    namedWindow("Display window", WINDOW_AUTOSIZE);

    // show the image
    imshow("Display window", image_src);

    plot_cost_graph(&image_gradient);
#endif

#ifdef TEST_DIJKSTRA
    std::srand((unsigned int)std::time(nullptr)); // use current time as seed for random generator


    stack< Point > test_points_stack;
    stack< Mat >   test_images_stack;
    stack< vector<Pixel_Node*>* > test_graphs_stack;

    int i, j, k, random_variable;
    int test_count = 0;

    for ( k = 0; k < 500; ++k) {

        auto start = std::chrono::system_clock::now();

        random_variable = std::rand();
        i = random_variable % rows;
        random_variable = std::rand();
        j = random_variable % cols;
        Point seed_point(i, j);

        random_variable = std::rand();
        i = random_variable % rows;
        random_variable = std::rand();
        j = random_variable % cols;
        Point dest_point(i, j);

        vector<Pixel_Node *> nodes_graph_for_seed;
        nodes_graph_for_seed = node_vector_original;
        cout << "seed x " << seed_point.x << " y " << seed_point.y << endl;
        cout << "dest x " << dest_point.x << " y " << dest_point.y << endl;

        minimum_cost_path_dijkstra(rows, cols, &seed_point, &nodes_graph_for_seed);
        test_graphs_stack.push(&nodes_graph_for_seed);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> running_seconds = end - start;
        cout << "dijkstra result " << ++test_count << " with " << running_seconds.count() << "seconds. " << endl;
    }
#endif

#ifdef PATH_TREE_TEST
    plot_path_tree(rows, cols, &nodes_graph_for_seed);

    Mat image_path_plot = image_src.clone();

    plot_path_tree_point_to_point(&seed_point, &current_mouse, &nodes_graph_for_seed, &image_path_plot);
#endif

    waitKey(20000);
    return 0;
}