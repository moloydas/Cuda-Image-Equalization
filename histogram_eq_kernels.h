// HISTOGRAM_EQ_KERNELS.h

#ifndef HISTOGRAM_EQ_KERNELS_H
#define HISTOGRAM_EQ_KERNELS_H

__global__ void histogramEqualization(const unsigned char* inputImage, unsigned char* outputImage, int imageSize, float* cdf);

__global__ void calculateHistogram(const unsigned char* inputImage, int imageSize, int* histogram);

__global__ void calculateCDF(int* histogram, int imageSize, float* cdf);

__global__ void calculateCDF_serial(int* histogram, int imageSize, float* cdf);

#endif  // HISTOGRAM_EQ_KERNELS_H
