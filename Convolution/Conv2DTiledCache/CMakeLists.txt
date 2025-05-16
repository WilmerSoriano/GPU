set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")

# Create main executable
add_executable(conv2d_tiled_cache_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp" "${SRC_PATH}/utils.cpp")
target_include_directories(conv2d_tiled_cache_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(conv2d_tiled_cache_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(conv2d_tiled_cache_test ${COMMON_LIBS})
add_dependencies(conv2d_tiled_cache_test libgputk)

# Set output directories
set_target_properties(conv2d_tiled_cache_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")