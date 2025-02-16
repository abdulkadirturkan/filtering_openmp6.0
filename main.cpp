#include <iostream>
#include <omp.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ctime>

#define max_threads 64

using namespace std;
using namespace cv;

void apply_mean_filter_openmp(const vector<vector<int>>& A, vector<vector<int>>& B, int H, int W) {
    int r = 2; // 5x5
    int filter_area = (2 * r + 1) * (2 * r + 1); // 25

//openMP 4.2
//#pragma omp parallel for shared(A, B, H, W, r) default(none) num_threads(max_threads)
//OpenMP 6.0
//#pragma omp parallel for shared(A, B, H, W, r) default(none) num_threads(max_threads) collapse(2) apply(reverse)
#pragma omp teams distribute parallel for simd shared(A, B, H, W, r) default(none) num_threads(max_threads)
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int sum = 0;
            int count = 0;

            for (int di = -r; di <= r; ++di) {
                for (int dj = -r; dj <= r; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && ni < H && nj >= 0 && nj < W) {
                        sum += A[ni][nj];
                        count++;
                    }
                }
            }

            B[i][j] = sum / count;
        }
    }
}

int main(int argc, char* argv[]) {
    //if (argc != 2) {
    //    cerr << "Usage: ./mean_filter_openmp <number_of_threads>" << endl;
    //    return -1;
    //}

    //int num_threads = atoi(argv[1]);

    if (max_threads <= 0) {
        cerr << "Invalid number of threads!" << endl;
        return -1;
    }

    omp_set_num_threads(max_threads);

    Mat image = imread("../Resources/pexels-pixasquare-1123982.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    int H = image.rows; // H
    int W = image.cols; // W

    vector<vector<int>> A(H, vector<int>(W));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            A[i][j] = image.at<uchar>(i, j);
        }
    }

    vector<vector<int>> B(H, vector<int>(W));

    clock_t start_time = clock();
    apply_mean_filter_openmp(A, B, H, W);
    clock_t end_time = clock();

    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Parallel time with " << max_threads << " threads: " << elapsed_time << " seconds" << endl;

    Mat output_image(H, W, CV_8U);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            output_image.at<uchar>(i, j) = B[i][j];
        }
    }

    imwrite("../Resources/output/pexels-pixasquare-1123982_openmp.png", output_image);
    cout << "Filtered image saved as filtered_pixasquare_openmp.png" << endl;

    return 0;
}
