set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")
file(GLOB DATAGEN_SOURCES "${SRC_PATH}/dataset_generator.cpp" "${SRC_PATH}/matmul_cpu.cpp")

# Create main executable
add_executable(tiledmatmul_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp")
target_include_directories(tiledmatmul_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(tiledmatmul_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(tiledmatmul_test ${COMMON_LIBS})
add_dependencies(tiledmatmul_test libgputk)

# Create datagen executable
add_executable(tiledmatmul_datagen ${DATAGEN_SOURCES})
target_include_directories(tiledmatmul_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(tiledmatmul_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(tiledmatmul_datagen ${COMMON_LIBS})
add_dependencies(tiledmatmul_datagen libgputk)

# Set output directories
set_target_properties(tiledmatmul_test tiledmatmul_datagen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")