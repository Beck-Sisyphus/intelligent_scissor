//
// Created by beck on 26/2/2018.
//

#include "plot.h"

#define DEBUG_POINT_TO_POINT

using namespace cv;
using namespace std;

cv::String cost_graph_directory = "../image/avatar_cost_graph.jpg";
cv::String path_tree_directory  = "../image/avatar_path_tree.jpg";
cv::String point_to_point_direc = "../image/point_to_point.jpg";
auto path_graph_curr_color = Scalar(255, 191, 0);
auto path_graph_prev_color = Scalar(127, 50, 0);
auto point_to_point_color  = Scalar(0, 127, 255);
auto point_to_path_color   = Scalar(0, 191, 255);

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

int plot_path_tree_point_to_point(Point* seed, Point* dest, vector<Pixel_Node*> *graph, Mat* image_plot)
{
    circle(*image_plot, *seed, 3, point_to_point_color, 2);
    circle(*image_plot, *dest, 3, point_to_point_color, 2);

    int index;
    index = dest->x * image_plot->cols + dest->y;
    Pixel_Node* dest_node = graph->data()[index];
    assert(dest_node != nullptr);

    index = seed->x * image_plot->cols + seed->y;
    Pixel_Node* seed_node = graph->data()[index];
    assert(seed_node != nullptr);

    Pixel_Node* curr_node = dest_node;
    Pixel_Node* prev_node;
    // Track back from the graph
    while ( !(*curr_node == *seed_node) )
    {
        prev_node = curr_node->prevNode;
        auto pointA = Point(curr_node->row, curr_node->col);
        auto pointB = Point(prev_node->row, prev_node->col);
        line(*image_plot, pointA, pointB, point_to_path_color, 2);

        curr_node = prev_node;
    }

#ifdef DEBUG_POINT_TO_POINT
    // Create the point to point path
    namedWindow("path point to point window", WINDOW_AUTOSIZE);
    imshow("path point to point window", *image_plot);

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