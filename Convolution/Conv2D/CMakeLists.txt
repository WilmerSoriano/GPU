set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")
file(GLOB DATAGEN_SOURCES "${SRC_PATH}/dataset_generator.cpp" "${SRC_PATH}/conv2d_cpu.cpp" "${SRC_PATH}/utils.cpp")

# Create main executable
add_executable(conv2d_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp" "${SRC_PATH}/utils.cpp")
target_include_directories(conv2d_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(conv2d_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(conv2d_test ${COMMON_LIBS})
add_dependencies(conv2d_test libgputk)

# Create datagen executable
add_executable(conv2d_datagen ${DATAGEN_SOURCES})
target_include_directories(conv2d_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(conv2d_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(conv2d_datagen ${COMMON_LIBS})
add_dependencies(conv2d_datagen libgputk)

# Set output directories
set_target_properties(conv2d_test conv2d_datagen
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")