cmake_minimum_required(VERSION 3.18)

project(yolo11_pose_npu LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    AI_SDK_ROOT
    "/home/radxa/repositories/ai-sdk"
    CACHE PATH
    "Path to the Allwinner/Radxa ai-sdk checkout"
)
set(
    VIPLITE_ROOT
    "${AI_SDK_ROOT}/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"
    CACHE PATH
    "Path to the A733 VIPLite v2.0 headers and libraries"
)

find_path(
    VIPLITE_INCLUDE_DIR
    NAMES vip_lite.h
    PATHS "${VIPLITE_ROOT}/inc"
    NO_DEFAULT_PATH
    REQUIRED
)
find_library(
    NBG_LINKER_LIBRARY
    NAMES NBGlinker
    PATHS "${VIPLITE_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED
)
find_library(
    VIP_HAL_LIBRARY
    NAMES VIPhal
    PATHS "${VIPLITE_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED
)

add_library(
    yolo11_pose_npu
    SHARED
    src/preprocess.cpp
    src/postprocess.cpp
    src/yolo11_pose_npu.cpp
)
target_include_directories(
    yolo11_pose_npu
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${VIPLITE_INCLUDE_DIR}"
)
target_compile_options(yolo11_pose_npu PRIVATE -Wall -Wextra -Wpedantic)
target_link_options(yolo11_pose_npu PRIVATE -Wl,--no-as-needed)
target_link_libraries(
    yolo11_pose_npu
    PRIVATE
    "${NBG_LINKER_LIBRARY}"
    "${VIP_HAL_LIBRARY}"
    dl
    m
    pthread
)
set_target_properties(
    yolo11_pose_npu
    PROPERTIES
    BUILD_RPATH "${VIPLITE_ROOT}"
    INSTALL_RPATH "${VIPLITE_ROOT}"
)

if(BUILD_TESTING)
    add_executable(
        yolo11_pose_postprocess_test
        tests/postprocess_test.cpp
        src/postprocess.cpp
    )
    target_include_directories(
        yolo11_pose_postprocess_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_compile_options(yolo11_pose_postprocess_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME yolo11_pose_postprocess COMMAND yolo11_pose_postprocess_test)

    add_executable(
        yolo11_pose_preprocess_test
        tests/preprocess_test.cpp
        src/preprocess.cpp
    )
    target_include_directories(
        yolo11_pose_preprocess_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_compile_options(yolo11_pose_preprocess_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME yolo11_pose_preprocess COMMAND yolo11_pose_preprocess_test)
endif()

install(TARGETS yolo11_pose_npu LIBRARY DESTINATION lib)
install(FILES include/yolo11_pose_npu.h DESTINATION include)
