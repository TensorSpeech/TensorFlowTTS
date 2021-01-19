cmake_minimum_required(VERSION 2.6)
PROJECT(TfliteTTS)

option(MAPPER "Processor select (supported BAKER or LJSPEECH)")
if (${MAPPER} STREQUAL "LJSPEECH")
    add_definitions(-DLJSPEECH)
elseif (${MAPPER} STREQUAL "BAKER")
    add_definitions(-DBAKER)
else ()
    message(FATAL_ERROR "MAPPER is only supported BAKER or LJSPEECH")
endif()

message(STATUS "MAPPER is selected: "${MAPPER})

include_directories(lib)
include_directories(lib/flatbuffers/include)
include_directories(src)

aux_source_directory(src DIR_SRCS)

SET(CMAKE_CXX_COMPILER "g++")

SET(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -Wl,--no-as-needed -ldl -pthread -fpermissive")

add_executable(demo demo/main.cpp ${DIR_SRCS})

find_library(tflite_LIB tensorflow-lite lib)

target_link_libraries(demo ${tflite_LIB})