# ============================================================
# CMakeLists.txt  (LENGKAP)
# ============================================================
cmake_minimum_required(VERSION 3.15)
project(ml_manual_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(USE_OPENBLAS "Gunakan OpenBLAS untuk matmul (opsional, ARM biasanya sudah ada build khusus)" OFF)
option(BUILD_PYTHON_BINDINGS "Build modul Python via pybind11" ON)

file(GLOB CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_library(ml_manual_cpp_core STATIC ${CORE_SOURCES})
target_include_directories(ml_manual_cpp_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(USE_OPENBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_compile_definitions(ml_manual_cpp_core PUBLIC USE_OPENBLAS)
        target_link_libraries(ml_manual_cpp_core PUBLIC ${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS tidak ditemukan, fallback ke matmul blocked manual")
    endif()
endif()

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Fallback umum di Termux/pip: cari cmake dir dari instalasi pybind11 via pip
        execute_process(
            COMMAND python3 -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    pybind11_add_module(ml_manual_cpp bindings/pybind_module.cpp)
    target_link_libraries(ml_manual_cpp PRIVATE ml_manual_cpp_core)
    target_include_directories(ml_manual_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

# ------------------------------------------------------------
# Cara build (Termux / Raspberry Pi / Linux umum):
#   pip install pybind11 --break-system-packages   # kalau belum ada
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENBLAS=ON
#   cmake --build build -j$(nproc)
#
# Hasil: build/ml_manual_cpp*.so — import dari Python lewat:
#   import sys; sys.path.append("build")
#   import ml_manual_cpp
# ------------------------------------------------------------