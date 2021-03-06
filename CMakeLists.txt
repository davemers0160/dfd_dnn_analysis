# ----------------------------------------------------------------------------
#  Root CMake file for DfD_DNN Project
# ----------------------------------------------------------------------------
#
# Example CMake build line for windows & VS2015
# windows w/cuda: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux w/o cuda: cmake -DUSE_AVX_INSTRUCTIONS=ON -DDLIB_USE_CUDA=OFF ..
#
#
# To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j10
#


cmake_minimum_required(VERSION 2.8.12)

PROJECT(DFD_DNN_Analysis_Project)
set(exe_name "dfd_dnn_an")

# This looks for an environment variable called PLATFORM.  This can be used to point
# cmake to various locations based on the individual machines that the code is complied on
set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})


if(${PLATFORM} STREQUAL "Laptop")
    set(COMMON_ROOT "D:/Common")
    set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
    set(DFD_COMMON_ROOT "D:/Projects/dfd_common")	
    include(D:/dlib-19.19/dlib/cmake)
elseif(${PLATFORM} STREQUAL "MainGear")
    set(_ECLIPSE_VERSION "3.8")
    set(COMMON_ROOT "~/Projects/Common")
    set(DLIB_CONTRIB "~/Projects/dlib-contrib")
    set(DFD_COMMON_ROOT "~/Projects/dfd_common")
    include(~/dlib-19.18/dlib/cmake)
elseif(${PLATFORM} STREQUAL "HPC")
    set(COMMON_ROOT "$HOME/Projects/Common")
    set(DLIB_CONTRIB "~/Projects/dlib-contrib")
    set(DFD_COMMON_ROOT "~/Projects/dfd_common")
    include(~/dlib/dlib/cmake)
else()
    set(COMMON_ROOT "D:/Common")
    set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
    set(DFD_COMMON_ROOT "D:/Projects/dfd_common")	
    include(D:/dlib-19.17/dlib/cmake)
endif()


# grab all of the *.c and *.cpp files in the src directory
file(GLOB SOURCES "src/*.c*"	
    "${COMMON_ROOT}/src/*.cpp"
    "${DLIB_CONTRIB}/src/*.cpp"
    )            
            
set(project_includes "include" "${COMMON_ROOT}/include" "${DLIB_CONTRIB}/include" "${DFD_COMMON_ROOT}/include")
          
message(" ")
message("--------------------------------------------------------")
message("DfD DNN Analysis Project Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

message("--------------------------------------------------------")
message("DfD DNN Analysis Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib)

if(WIN32)
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()
endif()

message(" ")
message(STATUS "Build Complete!")
message(" ")
