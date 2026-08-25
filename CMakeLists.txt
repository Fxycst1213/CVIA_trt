cmake_minimum_required(VERSION 3.16)
project(trt)

set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)
option(CVIA_PRODUCTION_BUILD "Build a stripped customer-delivery executable" OFF)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
enable_language(CUDA)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wfatal-errors -pthread -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC -w")

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
    ${PROJECT_SOURCE_DIR}/src/IRcamera
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

add_executable(trt ${srcs} )

if(CVIA_PRODUCTION_BUILD)
    # Do not embed the developer workspace path in __FILE__ strings or debug metadata.
    target_compile_options(trt PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-DNDEBUG;-fvisibility=hidden;-fvisibility-inlines-hidden;-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.;-fmacro-prefix-map=${PROJECT_SOURCE_DIR}=.>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-DNDEBUG;-Xcompiler=-fvisibility=hidden;-Xcompiler=-fvisibility-inlines-hidden>)
    target_link_options(trt PRIVATE -Wl,--strip-all)
endif()

target_link_libraries(trt ${CUDA_LIBRARIES})
target_link_libraries(trt nvinfer nvinfer_plugin)
target_link_libraries(trt cudart curand)
target_link_libraries(trt ${OpenCV_LIBS})
target_link_libraries(trt ${ZED_LIBS})
target_link_libraries(trt fftw3)
