cmake_minimum_required(VERSION 3.16)
project(cvia_mocap_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NOKOV_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NOKOV_SDK_LIBRARY
    "${NOKOV_SDK_ROOT}/lib/aarch64/libnokov_sdk.so")

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  message(FATAL_ERROR
    "The bundled NOKOV SDK is for NVIDIA Jetson/AArch64; current processor is "
    "${CMAKE_SYSTEM_PROCESSOR}.")
endif()
if(NOT EXISTS "${NOKOV_SDK_LIBRARY}")
  message(FATAL_ERROR "Missing NOKOV SDK library: ${NOKOV_SDK_LIBRARY}")
endif()

add_executable(MocapBridge MocapBridge.cpp)
target_include_directories(MocapBridge PRIVATE "${NOKOV_SDK_ROOT}/include")
target_link_libraries(MocapBridge PRIVATE "${NOKOV_SDK_LIBRARY}" pthread)

set_target_properties(MocapBridge PROPERTIES
  BUILD_RPATH "${NOKOV_SDK_ROOT}/lib/aarch64"
  INSTALL_RPATH "$ORIGIN/../lib/aarch64"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
