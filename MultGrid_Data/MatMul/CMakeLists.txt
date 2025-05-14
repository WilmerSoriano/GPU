set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")
file(GLOB DATAGEN_SOURCES "${SRC_PATH}/dataset_generator.cpp" "${SRC_PATH}/matmul_cpu.cpp")

# Create main executable
add_executable(matmul_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp")
target_include_directories(matmul_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(matmul_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(matmul_test ${COMMON_LIBS})
add_dependencies(matmul_test libgputk)

# Create datagen executable
add_executable(matmul_datagen ${DATAGEN_SOURCES})
target_include_directories(matmul_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(matmul_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(matmul_datagen ${COMMON_LIBS})
add_dependencies(matmul_datagen libgputk)

# Set output directories
set_target_properties(matmul_test matmul_datagen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")