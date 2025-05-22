cmake_minimum_required(VERSION 3.10)
project(Hand-volume-control)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Path of torch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)

# Path of opencv
find_package(OpenCV REQUIRED)

# Add libraries
add_library(volume_control src/volume_control.cpp include/volume_control.hpp)
add_library(detector src/detector.cpp include/detector.hpp)
add_library(utils src/utils.cpp include/utils.hpp)
add_library(model src/model.cpp include/model.hpp)

add_executable(hand_volume_control src/main.cpp)

# Link main.cpp with libraries
target_link_libraries(hand_volume_control
        volume_control
        detector
        utils
        model
        "${TORCH_LIBRARIES}"
        "${OpenCV_LIBS}"
        "/opt/homebrew/opt/libomp/lib/libomp.dylib"
)

set_property(TARGET hand_volume_control PROPERTY CXX_STANDARD 20)
set_property(TARGET hand_volume_control PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,/opt/homebrew/opt/libomp/lib")