#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "histogram_eq_kernels.h"

namespace py = pybind11;

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

// Wrapper function to perform histogram equalization
void histogramEqualizationWrapper(py::array_t<unsigned char> inputImage, py::array_t<unsigned char> outputImage, int width, int height) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Access input and output arrays
    auto inputPtr = inputImage.mutable_data();
    auto outputPtr = outputImage.mutable_data();

    const int imageSize = width * height;

    // Allocate memory for the image on the host
    unsigned char *h_inputImage = inputPtr;
    unsigned char *h_outputImage = outputPtr;

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    float *d_cdf;

    cudaMalloc((void **)&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_outputImage, imageSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_cdf, HISTOGRAM_SIZE * sizeof(float));

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

    // Copy the CDF back to host
    // float cdf[HISTOGRAM_SIZE];
    // cudaMemcpy(cdf, d_cdf, HISTOGRAM_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Set grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((imageSize + blockDim.x - 1) / blockDim.x);

    // Launch histogramEqualization kernel
    histogramEqualization<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, imageSize, d_cdf);

    // Copy the result back to host
    cudaMemcpy(h_outputImage, d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    printf("Total Cpp runtime calculation time: %f ms\n", totalTime);

    // Free allocated memory on the device
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_cdf);
    cudaFree(d_histogram);
}

// Bindings for Python
PYBIND11_MODULE(cuda_hist_eq, m) {
    m.doc() = "CUDA Histogram Equalization";

    m.def("histogram_equalization", &histogramEqualizationWrapper, "Perform histogram equalization using CUDA");
}

