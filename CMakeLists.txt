cmake_minimum_required(VERSION 3.30)

project(TerminalFluiSimulation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra -Wno-maybe-uninitialized")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(FluTe
    src/main.cpp
    src/img.cpp
)