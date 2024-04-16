cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(histogram_eq)
enable_language(CUDA)

# Set C++ Standard (optional)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV and CUDA libraries
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

add_library(cuda_hist_eq MODULE histogram_eq_wrapper.cu histogram_eq_kernels.cu)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Set CUDA compile flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

# Separate source files
file(GLOB_RECURSE CUDA_SOURCES histogram_eq_main.cu histogram_eq_kernels.cu)

# Add executable
add_executable(histogram_eq ${CUDA_SOURCES})

# Compile CUDA sources with NVCC
set_property(SOURCE ${CUDA_SOURCES} PROPERTY CUDA_COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})

target_link_libraries(cuda_hist_eq PRIVATE pybind11::module ${OpenCV_LIBS})
set_target_properties(cuda_hist_eq PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                           SUFFIX "${PYTHON_MODULE_EXTENSION}")