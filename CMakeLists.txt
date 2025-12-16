cmake_minimum_required(VERSION 3.16)
project(trt)

set(CMAKE_CUDA_ARCHITECTURES 87)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
enable_language(CUDA)


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_GEN_CODE "-gencode=arch=compute_87,code=sm_87")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

set(TENSORRT_INCLUDE_DIR     "/usr/include/aarch64-linux-gnu/")
set(TENSORRT_LIB_DIR     "/usr/lib/aarch64-linux-gnu/")

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(
    ${PROJECT_SOURCE_DIR}/src/preprocess
    ${PROJECT_SOURCE_DIR}/src/yolov8_detector
    ${PROJECT_SOURCE_DIR}/src/yolov8_classifier
    ${PROJECT_SOURCE_DIR}/src/yolov_pose
    ${PROJECT_SOURCE_DIR}/src/TensorRT/calibrator
    ${PROJECT_SOURCE_DIR}/src/TensorRT/CUDA_tools
    ${PROJECT_SOURCE_DIR}/src/TensorRT/logger
    ${PROJECT_SOURCE_DIR}/src/TensorRT/model
    ${PROJECT_SOURCE_DIR}/src/TensorRT/utils
    ${PROJECT_SOURCE_DIR}/src/time
    ${PROJECT_SOURCE_DIR}/src/lock
    ${PROJECT_SOURCE_DIR}/src/worker
    ${PROJECT_SOURCE_DIR}/src/ZEDX
    ${PROJECT_SOURCE_DIR}/src/communication
    ${PROJECT_SOURCE_DIR}/src/filter
    ${ZED_INCLUDE_DIRS}
)

link_directories(${TENSORRT_LIB_DIR})
link_directories(${ZED_LIBRARY_DIR}) 
link_libraries(nvinfer nvinfer_plugin nvonnxparser cuda cublas cudart cudnn)
link_libraries(${CUDA_LIBRARIES})
link_libraries(${OpenCV_LIBS})

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES})
endif()

link_libraries(stdc++fs)

file(GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(trt ${srcs} )

target_link_libraries(trt ${CUDA_LIBRARIES})
target_link_libraries(trt nvinfer nvinfer_plugin)
target_link_libraries(trt cudart curand)
target_link_libraries(trt ${OpenCV_LIBS})
target_link_libraries(trt ${ZED_LIBS})