cmake_minimum_required(VERSION 3.20)
project(vllm_scheduler_runtime VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SCHEDULER_BUILD_TESTS "Build GoogleTest suite" ON)

add_library(scheduler_runtime STATIC
  src/feature_extractor.cpp
  src/priority_queue.cpp
  src/inference_engine.cpp
)

target_include_directories(scheduler_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(scheduler_runtime PRIVATE /W4 /permissive-)
else()
  target_compile_options(scheduler_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(schedulerd src/main.cpp)
target_link_libraries(schedulerd PRIVATE scheduler_runtime)

if(SCHEDULER_BUILD_TESTS)
  enable_testing()
  add_executable(scheduler_tests tests/test_scheduler.cpp)
  target_link_libraries(scheduler_tests PRIVATE scheduler_runtime)
  add_test(NAME scheduler_tests COMMAND scheduler_tests)
endif()

install(TARGETS scheduler_runtime schedulerd
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
