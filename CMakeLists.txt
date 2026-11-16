cmake_minimum_required(VERSION 3.16)
project(trt)

set(CMAKE_CUDA_ARCHITECTURES 87)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
enable_language(CUDA)


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wfatal-errors -pthread -w")
set(CUDA_GEN_CODE "-gencode=arch=compute_87,code=sm_87")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xcompiler -fPIC -w ${CUDA_GEN_CODE}")

set(TENSORRT_INCLUDE_DIR     "/usr/include/aarch64-linux-gnu/")
set(TENSORRT_LIB_DIR     "/usr/lib/aarch64-linux-gnu/")

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
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
    ${PROJECT_SOURCE_DIR}/src/lstm
    ${PROJECT_SOURCE_DIR}/src/time
    ${PROJECT_SOURCE_DIR}/src/lock
    ${PROJECT_SOURCE_DIR}/src/worker
    ${PROJECT_SOURCE_DIR}/src/ZEDX
    ${PROJECT_SOURCE_DIR}/src/communication
    ${PROJECT_SOURCE_DIR}/src/params
    ${PROJECT_SOURCE_DIR}/src/algorithms
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
# trt uses the wts-nanhang TCP client; the standalone CVIA streamer uses a different API.
list(REMOVE_ITEM srcs "${PROJECT_SOURCE_DIR}/src/communication/frame_streamer.cpp")

add_executable(trt ${srcs} )

target_link_libraries(trt ${CUDA_LIBRARIES})
target_link_libraries(trt nvinfer nvinfer_plugin)
target_link_libraries(trt cudart curand)
target_link_libraries(trt ${OpenCV_LIBS})
target_link_libraries(trt ${ZED_LIBS})
target_link_libraries(trt fftw3)

# Offline dataset evaluation is opt-in and does not start application I/O.
set(keypoint_eval_srcs ${srcs})
list(REMOVE_ITEM keypoint_eval_srcs "${PROJECT_SOURCE_DIR}/src/main.cpp")
add_executable(keypoint_collector EXCLUDE_FROM_ALL tools/collect_keypoints.cpp ${keypoint_eval_srcs})
target_link_libraries(keypoint_collector ${CUDA_LIBRARIES} nvinfer nvinfer_plugin
    cudart curand ${OpenCV_LIBS} ${ZED_LIBS} fftw3)

add_executable(keypoint_refinement_eval EXCLUDE_FROM_ALL tools/refine_keypoints_sequence.cpp
    src/yolov_pose/keypoint_refiner.cpp)
set_property(TARGET keypoint_refinement_eval PROPERTY LINK_LIBRARIES ${OpenCV_LIBS})

add_executable(keypoint_refinement_tests EXCLUDE_FROM_ALL tests/test_keypoint_refiner.cpp
    src/yolov_pose/keypoint_refiner.cpp)
set_property(TARGET keypoint_refinement_tests PROPERTY LINK_LIBRARIES ${OpenCV_LIBS})
