cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(submodules/pybind11)

# python package ----------------------------------------------
project(DiffMITC3Impl)

pybind11_add_module(DiffMITC3Impl src/pybind.cpp)
target_include_directories(DiffMITC3Impl PUBLIC
        submodules/eigen
        submodules/pybind11/include
        # TODO : include Python.h in this directory
        /usr/include/python3.10)