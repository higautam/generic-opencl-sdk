cmake_minimum_required(VERSION 2.6)
project( GEN_OCL_SDK )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")

# We have custom written Find* modules now in the root source directory
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(OpenCL)

if(UNIX)
	set(CMAKE_CXX_FLAGS "-g3 -Wall -O0")
endif()

include_directories(${GEN_OCL_SDK_SOURCE_DIR}/include ${OPENCL_INCLUDE_DIRS})

#add_executable(AsyncDMA src/AsyncDMA/AsyncDMA.cpp)
#target_link_libraries(AsyncDMA ${OPENCL_LIBRARIES})
add_subdirectory(src)
