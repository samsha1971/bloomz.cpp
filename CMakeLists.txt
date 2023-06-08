cmake_minimum_required(VERSION 3.0.0)
project(bloom.cpp VERSION 0.1.0)

message(NOTICE "${CMAKE_COMPILER_IS_GNUCC}")

set (CMAKE_CXX_STANDARD 11)
IF(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DNDEBUG -fPIC -pthread -mfma -mf16c -mavx -mavx2")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -DNDEBUG -fPIC -pthread -mfma -mf16c -mavx -mavx2")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0")
    SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

set(EXECUTABLE_OUTPUT_PATH ../)

include(CTest)
enable_testing()

add_executable(main main.cpp ggml.c utils.cpp)
add_executable(quantize quantize.cpp ggml.c utils.cpp)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    set (ICONV_HOME /usr/local)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${ICONV_HOME}/include)
    target_link_libraries(main ${ICONV_HOME}/lib/libiconv.so)
    target_link_libraries(quantize ${ICONV_HOME}/lib/libiconv.so)
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_EXE_LINKER_FLAGS -static)
    set (ICONV_HOME C:/msys64/usr/local/mingw64)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${ICONV_HOME}/include)
    target_link_libraries(main ${ICONV_HOME}/lib/libiconv.a)
    target_link_libraries(quantize ${ICONV_HOME}/lib/libiconv.a)
else()
    message("current platform: unknown ")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
