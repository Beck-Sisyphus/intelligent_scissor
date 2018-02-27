//
// Created by beck on 26/2/2018.
//

#include "plot.h"

using namespace cv;
using namespace std;

cv::String cost_graph_directory = "../image/avatar_cost_graph.jpg";
cv::String path_tree_directory  = "../image/avatar_path_tree.jpg";

int plot_cost_graph(Mat* image_gradient)
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
                cout << "prev x: " << x << " y: " << y << endl;
                complete_path_tree.at<Vec3b>(x, y)[0] = 255;
                complete_path_tree.at<Vec3b>(x, y)[1] = 255;
                complete_path_tree.at<Vec3b>(x, y)[2] = 0;
                y = 2 * prev->col + curr->col + 1;
                x = 2 * prev->row + curr->row + 1;
                cout << "curr x: " << x << " y: " << y << endl;
                complete_path_tree.at<Vec3b>(x, y)[0] = 127;
                complete_path_tree.at<Vec3b>(x, y)[1] = 127;
                complete_path_tree.at<Vec3b>(x, y)[2] = 0;
            }
        }
    }
    // Create a window
    namedWindow("gradient window", WINDOW_AUTOSIZE);
    imshow("gradient window", complete_path_tree);

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

    return 1;
}