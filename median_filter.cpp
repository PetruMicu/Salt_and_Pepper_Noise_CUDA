#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

// Function to compute the median of a set of values
unsigned char findMedian(unsigned char* arr, int size) {
    sort(arr, arr + size);
    return arr[size / 2];
}

// CPU-based median filter function
void applyMedianFilterCPU(const Mat &src, Mat &dst, int kernelSize = 3) {
    int pad = kernelSize / 2;
    dst = src.clone();
    for (int i = pad; i < src.rows - pad; i++) {
        for (int j = pad; j < src.cols - pad; j++) {
            for (int c = 0; c < 3; c++) {  // For each channel (BGR)
                unsigned char window[kernelSize * kernelSize];
                int k = 0;

                // Extract the neighborhood for the current pixel
                for (int m = -pad; m <= pad; m++) {
                    for (int n = -pad; n <= pad; n++) {
                        window[k++] = src.at<Vec3b>(i + m, j + n)[c];
                    }
                }

                // Find the median in the window
                dst.at<Vec3b>(i, j)[c] = findMedian(window, kernelSize * kernelSize);
            }
        }
    }
}

int main(int argc, char** argv) {
    // Check if image path is passed as a command-line argument
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // Get image path from command-line argument
    string imagePath = argv[1];

    // Load image
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Could not open or find the image at path: " << imagePath << endl;
        return -1;
    }

    // Apply median filter using CPU
    Mat cpuFilteredImage;
    auto startCPU = chrono::high_resolution_clock::now();
    applyMedianFilterCPU(image, cpuFilteredImage, 3);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> durationCPU = endCPU - startCPU;
    cout << "CPU processing time: " << durationCPU.count() * 1000.0 << " ms" << endl;

    // Show the images
//    imshow("Original Image", image);
//    imshow("Filtered Image (CPU)", cpuFilteredImage);
    imwrite("../cpu_filtered_image.jpg", cpuFilteredImage);
    waitKey(0);

    return 0;
}
