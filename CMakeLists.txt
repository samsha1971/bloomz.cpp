cmake_minimum_required(VERSION 3.0.0)
project(bloom.cpp VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(main main.cpp ggml.c utils.cpp)

add_executable(quantize quantize.cpp ggml.c utils.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
