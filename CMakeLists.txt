cmake_minimum_required(VERSION 3.0.0)
project(MNIST_Recog VERSION 0.1.0)

add_subdirectory("CustomLibrary")
link_libraries(CustomLibrary)

add_executable(${PROJECT_NAME} src/main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
