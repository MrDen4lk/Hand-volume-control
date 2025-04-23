cmake_minimum_required(VERSION 3.10)
project(Hand-volume-control)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Path of torch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)

# Path of opencv
find_package(OpenCV REQUIRED)

# Source files
file(GLOB SRC_FILES src/*.cpp)
add_executable(hand_volume_control ${SRC_FILES})

# Link torch and libomp
target_link_libraries(hand_volume_control
        "${TORCH_LIBRARIES}"
        "${OpenCV_LIBS}"
        "/opt/homebrew/opt/libomp/lib/libomp.dylib"
)

set_property(TARGET hand_volume_control PROPERTY CXX_STANDARD 20)
set_property(TARGET hand_volume_control PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,/opt/homebrew/opt/libomp/lib")