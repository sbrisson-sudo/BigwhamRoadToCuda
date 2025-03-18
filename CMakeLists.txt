cmake_minimum_required(VERSION 3.16)

project(Hmat_cuda VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# find_package(BLAS REQUIRED) // in inside loop

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

add_subdirectory(il)

include_directories(
    ${CBLAS_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14)

add_executable(gpu_hmat_matvec src/gpu_hmat_matvec.cpp)
target_link_libraries(gpu_hmat_matvec 
    InsideLoop 
    z 
    dl pthread rt # https://stackoverflow.com/questions/62962539/linking-error-for-cuda-separate-compilation-mode-and-static-linking-of-cudart
    ${BLAS_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cudart_static_LIBRARY}
    )
target_include_directories(gpu_hmat_matvec PUBLIC src il cnpy )

message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
message(STATUS "CUDA_CUSPARSE_LIBRARIES: ${CUDA_cusparse_LIBRARY}")
message(STATUS "CUDA_cudart_static_LIBRARY: ${CUDA_cudart_static_LIBRARY}")
