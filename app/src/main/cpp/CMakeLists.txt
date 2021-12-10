cmake_minimum_required(VERSION 3.10.2)

project("smarthome")

add_library(GLES_mali STATIC IMPORTED)

add_library(
        JNIDriver
        SHARED
        JNIDriver.c)

find_library(log-lib log)

find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
        JNIDriver
        C:/Users/parlab/Documents/OpenCL_lib_and_include/lib/libGLES_mali.so
        ${log-lib} ${jnigraphics-lib})

target_include_directories(JNIDriver PRIVATE src/main/cpp)