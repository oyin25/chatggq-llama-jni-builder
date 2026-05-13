cmake_minimum_required(VERSION 3.22.1)

project(chatggq_llama_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(
        llama_jni
        SHARED
        llama_jni.cpp
)

target_include_directories(
        llama_jni
        PRIVATE
        llama.cpp/include
        llama.cpp/ggml/include
        llama.cpp/ggml/src
)

target_link_libraries(
        llama_jni
        PRIVATE
        llama
        android
        log
)
