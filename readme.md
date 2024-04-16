# Requirements

- install Opencv
- pip install pybind11


# Code Structure

### histogram_eq_kernels.h

All the kernels are declared here

### histogram_eq_kernels.cu

All the kernels are defined here

### histogram_eq_wrapper.cu

wrapper for python bindings

### histogram_eq_main.cu

main cpp code to validate the implementation. prints the timings of the GPU kernel and compares it to OpenCV implementation. My implementation should have a lower runtime.

### test_hist_eq.py

main py code to test bindings. prints the timings of the GPU kernel and compares it to OpenCV implementation. 
Usually OpenCV wins here. Calling cpp functions from python is very heavy