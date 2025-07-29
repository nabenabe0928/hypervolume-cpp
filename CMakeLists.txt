cmake_minimum_required(VERSION 3.11)  # FetchContent is available in 3.11+
set(CMAKE_CXX_STANDARD 17)
project(hvcpp)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(hvcpp hv.cpp)
