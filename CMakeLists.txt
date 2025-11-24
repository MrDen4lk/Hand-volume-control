cmake_minimum_required(VERSION 3.10)
project(Hand-volume-control)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Path of opencv
find_package(OpenCV REQUIRED)

find_library(COREAUDIO_LIBRARY CoreAudio)
find_library(AUDIOTOOLBOX_LIBRARY AudioToolbox)

# Path of onnxruntime
set(BREW_ROOT "/opt/homebrew")
include_directories(${BREW_ROOT}/include/onnxruntime)
link_directories(${BREW_ROOT}/lib)

# Add libraries
add_library(volume_control
        src/volume_control.cpp
        include/volume_control.hpp
)
target_link_libraries(volume_control
        "${OpenCV_LIBS}"
        "${COREAUDIO_LIBRARY}"
        "${AUDIOTOOLBOX_LIBRARY}"
)

add_library(hand_models
        src/yolo_model.cpp
        include/yolo_model.hpp
        include/hand_model_base.hpp
        src/convnext_base_model.cpp
        include/convnext_base_model.hpp
)
target_link_libraries(hand_models
        "${OpenCV_LIBS}"
        onnxruntime
)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(hand_volume_control src/main.cpp)

# Link files with libraries
target_link_libraries(hand_volume_control
        volume_control
        hand_models
        "${OpenCV_LIBS}"
        onnxruntime
        "${BREW_ROOT}/opt/libomp/lib/libomp.dylib"
)

set_property(TARGET hand_volume_control PROPERTY CXX_STANDARD 20)
set_property(TARGET hand_volume_control PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,${BREW_ROOT}/opt/libomp/lib -Wl,-rpath,${BREW_ROOT}/lib")