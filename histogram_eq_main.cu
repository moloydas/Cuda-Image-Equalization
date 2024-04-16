#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "histogram_eq_kernels.h"

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

int main() {
    // Read the image using OpenCV
    cv::Mat originalImage = cv::imread("../kodim08_grayscale.png", cv::IMREAD_GRAYSCALE);

    if (originalImage.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    //////////////////////////////////////////////////////////////////////
    // CPU implementation
    //////////////////////////////////////////////////////////////////////
    cudaEvent_t startCpu, stopCpu;
    cudaEventCreate(&startCpu);
    cudaEventCreate(&stopCpu);

    cv::Mat originalImageCPU;
    cv::Mat equalizedImageCpu;
    originalImageCPU = originalImage.clone();

    // Record the startCpu time
    cudaEventRecord(startCpu, 0);

    // Perform histogram equalization on the CPU
    cv::equalizeHist(originalImage, equalizedImageCpu);
    cudaEventRecord(stopCpu, 0);
    cudaEventSynchronize(stopCpu);
    
    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startCpu, stopCpu);

    // Print the elapsed time
    printf("CPU Elapsed Time: %f ms\n", elapsedTime);

    // Save the histogram-equalized image using OpenCV
    cv::imwrite("equalized_image_cpu.png", equalizedImageCpu);

    // Destroy CUDA events
    cudaEventDestroy(startCpu);
    cudaEventDestroy(stopCpu);
    //////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////
    // GPU implementation
    //////////////////////////////////////////////////////////////////////

    // Specify image dimensions
    const int width = originalImage.cols;
    const int height = originalImage.rows;
    const int imageSize = width * height;

    // Allocate memory for the image on the host
    unsigned char *h_inputImage = originalImage.data;
    unsigned char *h_outputImage = new unsigned char[imageSize];

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    float *d_cdf;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_outputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_cdf, HISTOGRAM_SIZE * sizeof(float));

    /////////////////////////////////////////////////////////////////////////
    // Start the timing here for the GPU implementation which include memcpy
    /////////////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);
    // Copy the input image to the device
    cudaMemcpy(d_inputImage, h_inputImage, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int *d_histogram;
    cudaMalloc((void **)&d_histogram, HISTOGRAM_SIZE * sizeof(int));

    // Set grid and block dimensions for histogram calculation
    dim3 histogramBlockDim(BLOCK_SIZE);
    dim3 histogramGridDim((imageSize + histogramBlockDim.x - 1) / histogramBlockDim.x);

    // Launch histogram calculation kernel
    calculateHistogram<<<histogramGridDim, histogramBlockDim>>>(d_inputImage, imageSize, d_histogram);

    // Set grid and block dimensions for CDF calculation
    dim3 cdfBlockDim(1);
    dim3 cdfGridDim(1);

    // Launch CDF calculation kernel
    calculateCDF_serial<<<cdfGridDim, cdfBlockDim>>>(d_histogram, imageSize, d_cdf);

    // // Copy the CDF back to host
    // float cdf[HISTOGRAM_SIZE];
    // cudaMemcpy(cdf, d_cdf, HISTOGRAM_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Set grid and block dimensions
    dim3 histEqBlockDim(BLOCK_SIZE);
    dim3 histEqGridDim((imageSize + histEqBlockDim.x - 1) / histEqBlockDim.x);

    // Launch histogramEqualization kernel
    histogramEqualization<<<histEqGridDim, histEqBlockDim>>>(d_inputImage, d_outputImage, imageSize, d_cdf);

    // Copy the result back to host
    cudaMemcpy(h_outputImage, d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    /////////////////////////////////////////////////////////////////////////
    // Stop the timing here for the GPU implementation which include memcpy
    /////////////////////////////////////////////////////////////////////////

    // Measure total execution time
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    printf("GPU Elapsed Time: %f ms\n", totalTime);

    // Save the histogram-equalized image using OpenCV
    cv::Mat equalizedImage(height, width, CV_8UC1, h_outputImage);
    cv::imwrite("equalized_image_gpu.png", equalizedImage);

    // Free allocated memory on the device
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_cdf);
    cudaFree(d_histogram);

    return 0;
}