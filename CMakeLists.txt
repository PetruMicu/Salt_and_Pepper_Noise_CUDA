cmake_minimum_required(VERSION 3.26)
#project(CUDA_NoiseFiltering CUDA)
project(CUDA_NoiseFiltering LANGUAGES CXX CUDA)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)

# Add the executable
add_executable(CUDA_Median_Filter median_filter.cu)
set_target_properties(CUDA_Median_Filter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Add the executable
add_executable(CPU_Median_Filter median_filter.cpp)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries to the target
target_link_libraries(CUDA_Median_Filter ${OpenCV_LIBS})

# Link OpenCV libraries to the target
target_link_libraries(CPU_Median_Filter ${OpenCV_LIBS})