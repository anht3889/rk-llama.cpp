# RKNPU2 backend for the RK3588 NPU, Linux AArch64 only

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "RKNPU2: GGML_RKNPU2 is only supported on Linux; target system is ${CMAKE_SYSTEM_NAME}")
endif()

if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "RKNPU2: GGML_RKNPU2 is only supported on AArch64; target architecture is ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if (NOT RKNPU2_SDK_ROOT)
    message(FATAL_ERROR "RKNPU2: GGML_RKNPU2 requires -DRKNPU2_SDK_ROOT=/path/to/runtime/Linux")
endif()

find_path(RKNPU2_INCLUDE_DIR
    NAMES rknn_matmul_api.h
    HINTS ${RKNPU2_SDK_ROOT}/librknn_api/include
    NO_DEFAULT_PATH)

find_library(RKNPU2_LIBRARY
    NAMES rknnrt
    HINTS ${RKNPU2_SDK_ROOT}/librknn_api/aarch64
    NO_DEFAULT_PATH)

if (NOT RKNPU2_INCLUDE_DIR)
    message(FATAL_ERROR "RKNPU2: rknn_matmul_api.h not found under ${RKNPU2_SDK_ROOT}/librknn_api/include")
endif()

if (NOT RKNPU2_LIBRARY)
    message(FATAL_ERROR "RKNPU2: librknnrt not found under ${RKNPU2_SDK_ROOT}/librknn_api/aarch64")
endif()

message(STATUS "RKNPU2 SDK root:    ${RKNPU2_SDK_ROOT}")
message(STATUS "RKNPU2 include dir: ${RKNPU2_INCLUDE_DIR}")
message(STATUS "RKNPU2 library:     ${RKNPU2_LIBRARY}")

ggml_add_backend_library(ggml-rknpu2
                         ggml-rknpu2.cpp
                        )

target_include_directories(ggml-rknpu2 PRIVATE ${RKNPU2_INCLUDE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(ggml-rknpu2 PRIVATE ${RKNPU2_LIBRARY} Threads::Threads)
