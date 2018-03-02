//
// Created by beck on 26/2/2018.
//

#include "plot.h"

//#define DEBUG_POINT_TO_POINT

using namespace cv;
using namespace std;

String cost_graph_directory = "../image/avatar_cost_graph.jpg";
String path_tree_directory  = "../image/avatar_path_tree.jpg";
String point_to_point_direc = "../image/point_to_point.jpg";
String plot_window_name = "main window";
auto path_graph_curr_color = Scalar(255, 191, 0);
auto path_graph_prev_color = Scalar(127, 50, 0);
auto point_to_point_color  = Scalar(0, 127, 255);
auto point_to_path_color   = Scalar(0, 191, 255);
auto pixel_node_color      = Scalar(0,  0,  0);


int plot_cost_graph(Mat *image_gradient)
{
    // Create a window
    namedWindow("gradient window", WINDOW_AUTOSIZE);
    imshow("gradient window", *image_gradient);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    try {
        imwrite(cost_graph_directory, *image_gradient, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 0;
    }
    fprintf(stdout, "Saved jpeg file for cost graph.\n");
    return 1;
}

/**
 * Plot the complete path tree
 * @param rows
 * @param cols
 * @param graph
 * @return
 */
int plot_path_tree(int rows, int cols, vector<Pixel_Node*> *graph)
{
    auto complete_path_tree = Mat( rows * 3, cols * 3, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            Pixel_Node* curr = graph->data()[index];
            Pixel_Node* prev = curr->prevNode;
            if (prev != NULL)
            {
                // Draw one point to another
                int x, y;
                x = 3 * i + 1 + prev->col - curr->col;
                y = 3 * j + 1 + prev->row - curr->row;
//                cout << "prev x: " << x << " y: " << y << endl;
                complete_path_tree.at<Vec3b>(x, y)[0] = (uchar)path_graph_curr_color[0];
                complete_path_tree.at<Vec3b>(x, y)[1] = (uchar)path_graph_curr_color[1];
                complete_path_tree.at<Vec3b>(x, y)[2] = (uchar)path_graph_curr_color[2];
                y = 2 * prev->col + curr->col + 1;
                x = 2 * prev->row + curr->row + 1;
//                cout << "curr x: " << x << " y: " << y << endl;
                complete_path_tree.at<Vec3b>(x, y)[0] = (uchar)path_graph_prev_color[0];
                complete_path_tree.at<Vec3b>(x, y)[1] = (uchar)path_graph_prev_color[1];
                complete_path_tree.at<Vec3b>(x, y)[2] = (uchar)path_graph_prev_color[2];
            }
        }
    }
    // Create a window
    namedWindow("path tree window", WINDOW_AUTOSIZE);
    imshow("path tree window", complete_path_tree);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    try {
        imwrite(path_tree_directory, complete_path_tree, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 0;
    }
    fprintf(stdout, "Saved jpeg for path tree.\n");
    return 1;
}

//int plot_path_tree_complete(int rows, int cols, vector<Pixel_Node*> *graph, int x, int y, Mat* image_src)
//{
//    auto complete_path_tree = Mat( rows, cols, CV_8UC3, Scalar(255, 255, 255));
//    complete_path_tree = image_src->clone();
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            int index = i * cols + j;
//            Pixel_Node* curr = graph->data()[index];
//            Pixel_Node* prev = curr->prevNode;
//            if (prev != NULL)
//            {
//                // Draw one point to another
//                int x, y;
//                x = 3 * i + 1 + prev->col - curr->col;
//                y = 3 * j + 1 + prev->row - curr->row;
////                cout << "prev x: " << x << " y: " << y << endl;
//                complete_path_tree.at<Vec3b>(x, y)[0] = (uchar)path_graph_curr_color[0];
//                complete_path_tree.at<Vec3b>(x, y)[1] = (uchar)path_graph_curr_color[1];
//                complete_path_tree.at<Vec3b>(x, y)[2] = (uchar)path_graph_curr_color[2];
//                y = 2 * prev->col + curr->col + 1;
//                x = 2 * prev->row + curr->row + 1;
////                cout << "curr x: " << x << " y: " << y << endl;
//                complete_path_tree.at<Vec3b>(x, y)[0] = (uchar)path_graph_prev_color[0];
//                complete_path_tree.at<Vec3b>(x, y)[1] = (uchar)path_graph_prev_color[1];
//                complete_path_tree.at<Vec3b>(x, y)[2] = (uchar)path_graph_prev_color[2];
//            }
//        }
//    }
//    // Create a window
//    namedWindow("path tree window", WINDOW_AUTOSIZE);
//    imshow("path tree window", complete_path_tree);
//
//    vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//    compression_params.push_back(95);
//    try {
//        imwrite(path_tree_directory, complete_path_tree, compression_params);
//    }
//    catch (runtime_error& ex) {
//        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//        return 0;
//    }
//    fprintf(stdout, "Saved jpeg for path tree.\n");
//    return 1;
//}
//

/**
 * Extend the image to pixel nodes
 * @param rows
 * @param cols
 * @param image_src
 * @return
 */
int plot_pixel_node(int rows, int cols, Mat* image_src)
{
    auto complete_pixel_node = Mat( rows * 3, cols * 3, CV_8UC3, pixel_node_color);
    int i, j, x, y;
    for ( i = 0; i < rows; ++i) {
        for ( j = 0; j < cols; ++j) {
            x = 3 * i + 1;
            y = 3 * j + 1;
            complete_pixel_node.at<Vec3b>(x, y)[0] = image_src->at<Vec3b>(i, j)[0];
            complete_pixel_node.at<Vec3b>(x, y)[1] = image_src->at<Vec3b>(i, j)[1];
            complete_pixel_node.at<Vec3b>(x, y)[2] = image_src->at<Vec3b>(i, j)[2];
        }
    }
    // Create a window
    namedWindow("pixel node window", WINDOW_AUTOSIZE);
    imshow("pixel node window", complete_pixel_node);
}



/**
 * Test case to calculate the countor from one point to another
 * @input seed
 * @input dest
 * @input graph
 * @output image_plot
 * @return
 */
int plot_path_tree_point_to_point(Point* seed, Point* dest, vector<Pixel_Node*> *graph, Mat* image_plot)
{
    int index;
    Pixel_Node *dest_node, *seed_node, *curr_node, *prev_node;
    index = dest->x * image_plot->cols + dest->y;
    dest_node = graph->data()[index];
    assert(dest_node != nullptr);

    index = seed->x * image_plot->cols + seed->y;
    seed_node = graph->data()[index];
    assert(seed_node != nullptr);

    curr_node = dest_node;

//    cout << "dest  of the node in plot " << dest_node->row << " col "<< dest_node->col << endl;
//    cout << "start of the node in plot " << curr_node->row << " col "<< curr_node->col << endl;

    // Track back from the graph
    while ( curr_node != nullptr && curr_node->prevNode != nullptr &&
            !(curr_node->row == seed_node->row && curr_node->col == seed_node->col))
    {
        prev_node = curr_node->prevNode;
        // Flip pixels in here too
        auto pointA = Point(curr_node->col, curr_node->row);
        auto pointB = Point(prev_node->col, prev_node->row);
        line(*image_plot, pointA, pointB, point_to_path_color, 2);

        curr_node = prev_node;
    }
//    cout << "end   of the node in plot row " << prev_node->row << " col "<< prev_node->col << endl;

#ifdef DEBUG_POINT_TO_POINT
    imshow(plot_window_name, *image_plot);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    try {
        imwrite(point_to_point_direc, *image_plot, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 0;
    }
    fprintf(stdout, "Saved jpeg for path tree.\n");
#endif

    return 1;
}