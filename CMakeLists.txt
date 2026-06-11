cmake_minimum_required(VERSION 3.16)

project(TAREA_2_2_2)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

include_directories(include)

add_library(tarea_2_2_2_libs
    src/ED.cpp
    src/PSO.cpp
    src/SHADE.cpp
    src/CMSA.cpp
    src/MLP.cpp
    src/utils.cpp
)

# Crear ejecutable
add_executable(main_exec main/main_MLP.cpp)

target_link_libraries(main_exec tarea_2_2_2_libs)

target_link_libraries(main_exec
    tarea_2_2_2_libs
    OpenMP::OpenMP_CXX
)