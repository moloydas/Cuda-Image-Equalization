#include "histogram_eq_kernels.h"

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

// CUDA kernel for histogram calculation
__global__ void calculateHistogram(const unsigned char* inputImage, int imageSize, int* histogram) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory for histogram
    __shared__ int partialHistogram[HISTOGRAM_SIZE];

    if(threadIdx.x < HISTOGRAM_SIZE){
        partialHistogram[threadIdx.x] = 0;
    }
    __syncthreads();

    if (tid < imageSize) {
        atomicAdd(&partialHistogram[inputImage[tid]], 1);
    }

    __syncthreads();

    if(threadIdx.x < HISTOGRAM_SIZE){
        atomicAdd(&histogram[threadIdx.x], partialHistogram[threadIdx.x]);
    }
}

// CDF calculation
__global__ void calculateCDF_serial(int* histogram, int imageSize, float* cdf){
    int i = 0;
    float cum_sum = 0.0f;
    while(i < HISTOGRAM_SIZE){
        cdf[i] = cum_sum + histogram[i];
        cum_sum += histogram[i];
        cdf[i] /= imageSize;
        i++;
    }
}

// CUDA kernel for CDF calculation
__global__ void calculateCDF(int* histogram, int imageSize, float* cdf) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory for CDF
    __shared__ float partialCDF[HISTOGRAM_SIZE];

    if(threadIdx.x < HISTOGRAM_SIZE){
        partialCDF[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculate CDF in each block
    for (int i = 0; i <= tid; ++i) {
        partialCDF[tid] += histogram[i];
    }

    __syncthreads();

    // Normalize CDF by the total number of pixels
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        cdf[i] = partialCDF[i] / imageSize;
    }
}

// CUDA kernel for global histogram equalization
__global__ void histogramEqualization(const unsigned char* inputImage, unsigned char* outputImage, int imageSize, float* cdf) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Apply histogram equalization to the input image
    outputImage[tid] = static_cast<unsigned char>(255.0f * (cdf[inputImage[tid]]));
}
