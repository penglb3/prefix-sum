cmake_minimum_required(VERSION 3.18)
project(PrefexSum LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC") # MSVC doesnt recognize these flags.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-volatile -Wno-address-of-packed-member")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(ExternalProject)
include(ProcessorCount)
ProcessorCount(N)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CTest)
include(CheckCXXCompilerFlag)
include(FindOpenMP)
include(ClangFormat)

if(NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
# valgrind has problem understanding certain native instructions, so don't use it in Debug.
set(USE_NATIVE_INST "Yes")
if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(USE_NATIVE_INST "No")
endif()
if(COMPILER_SUPPORTS_MARCH_NATIVE AND USE_NATIVE_INST MATCHES "Yes")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

CHECK_CXX_COMPILER_FLAG("-flto" COMPILER_SUPPORTS_FLTO)
if(COMPILER_SUPPORTS_FLTO)
  if (CMAKE_BUILD_TYPE STREQUAL Release)    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
  endif()
endif()

if(OpenMP_FOUND)
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  endif()
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-path=\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1\"")

if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-but-set-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines-ts -stdlib=libc++ -Wunused-but-set-variable")
endif ()

set(PS_HEADERS
  src/common.h
)

set(PS_SOURCES
  src/cuda.cu
  src/omp.cc
)

include_directories(src)

add_executable(prefix-sum src/main.cc ${PS_SOURCES})

find_program(clang_format_executable clang-format)
if (CLANGFORMAT_EXECUTABLE OR clang_format_executable)
  message("Clang format found")
  clangformat_setup(
    ${PRFAS_SOURCES}
    src/main.cc
    ${PRFAS_HEADERS}
  )
endif()
