#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;

#define BLOCK_SIZE 16

__device__ void swap(unsigned char &a, unsigned char &b) {
    unsigned char temp = a;
    a = b;
    b = temp;
}

// CUDA Kernel for Median Filtering
__global__ void applyMedianFilter(const uchar4* src, uchar4* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Shared memory for the neighborhood window
        __shared__ uchar4 shared_mem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

        int lx = threadIdx.x + 1;
        int ly = threadIdx.y + 1;

        shared_mem[ly][lx] = src[y * width + x];

        // Load halo pixels
        if (threadIdx.x < 1 && x > 0) {
            shared_mem[ly][lx - 1] = src[y * width + x - 1];
        }
        if (threadIdx.x >= blockDim.x - 1 && x < width - 1) {
            shared_mem[ly][lx + 1] = src[y * width + x + 1];
        }
        if (threadIdx.y < 1 && y > 0) {
            shared_mem[ly - 1][lx] = src[(y - 1) * width + x];
        }
        if (threadIdx.y >= blockDim.y - 1 && y < height - 1) {
            shared_mem[ly + 1][lx] = src[(y + 1) * width + x];
        }

        __syncthreads();

        // Apply median filter
        if (lx > 0 && lx < blockDim.x + 1 && ly > 0 && ly < blockDim.y + 1) {
            uchar windowR[9], windowG[9], windowB[9];
            int k = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    uchar4 pixel = shared_mem[ly + i][lx + j];
                    windowB[k] = pixel.x;
                    windowG[k] = pixel.y;
                    windowR[k] = pixel.z;
                    k++;
                }
            }

            // Sort the arrays to find the median
            for (int i = 0; i < 9; i++) {
                for (int j = i + 1; j < 9; j++) {
                    if (windowB[i] > windowB[j]) swap(windowB[i], windowB[j]);
                    if (windowG[i] > windowG[j]) swap(windowG[i], windowG[j]);
                    if (windowR[i] > windowR[j]) swap(windowR[i], windowR[j]);
                }
            }

            dst[y * width + x] = make_uchar4(windowB[4], windowG[4], windowR[4], 255); // Median
        }
    }
}

int main(int argc, char** argv) {
    // Initialize CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
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

    // Convert to RGBA
    Mat imageRGBA;
    cvtColor(image, imageRGBA, COLOR_BGR2RGBA);

    // Upload image to GPU
    cuda::GpuMat d_src(imageRGBA);
    cuda::GpuMat d_dst(imageRGBA.size(), imageRGBA.type());

    // Define grid and block dimensions
    const int blockSize = BLOCK_SIZE;
    dim3 block(blockSize, blockSize);
    dim3 grid((d_src.cols + block.x - 1) / block.x, (d_src.rows + block.y - 1) / block.y);

    // Start timing before kernel launch
    cudaEventRecord(start);

    // Launch kernel
    applyMedianFilter<<<grid, block>>>(d_src.ptr<uchar4>(), d_dst.ptr<uchar4>(), d_src.cols, d_src.rows);

    // Stop timing after kernel execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "CUDA kernel execution time: " << milliseconds << " ms" << endl;

    // Download result from GPU
    Mat filteredImage;
    d_dst.download(filteredImage);

    // Convert back to BGR
    cvtColor(filteredImage, filteredImage, COLOR_RGBA2BGR);

    // Display images
//    imshow("Original Image", image);
//    imshow("Filtered Image", filteredImage);
    imwrite("../cuda_filtered_image.jpg", filteredImage);
    waitKey(0);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
