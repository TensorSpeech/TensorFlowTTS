cmake_minimum_required(VERSION 2.6)
PROJECT(TfliteTTS)

include_directories(lib)
include_directories(lib/flatbuffers/include)
include_directories(src)

aux_source_directory(src DIR_SRCS)

SET(CMAKE_CXX_COMPILER "g++")

SET(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -Wl,--no-as-needed -ldl -pthread -fpermissive")

add_executable(demo demo/main.cpp ${DIR_SRCS})

find_library(tflite_LIB tensorflow-lite lib)

target_link_libraries(demo ${tflite_LIB})