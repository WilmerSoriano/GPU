set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")
file(GLOB DATAGEN_SOURCES "${SRC_PATH}/dataset_generator.cpp" "${SRC_PATH}/blur_cpu.cpp")

# Create main executable
add_executable(blur ${CUDA_SOURCES} "${SRC_PATH}/main.cpp")
target_include_directories(blur PRIVATE
    ${COMMON_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(blur PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(blur
    ${COMMON_LIBS}
    ${OpenCV_LIBS}
)
add_dependencies(blur libgputk)

# Create datagen executable
add_executable(blur_datagen ${DATAGEN_SOURCES})
target_include_directories(blur_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(blur_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(blur_datagen ${COMMON_LIBS})
add_dependencies(blur_datagen libgputk)

# Create test executable
add_executable(blur_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp")
target_include_directories(blur_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(blur_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(blur_test ${COMMON_LIBS})
add_dependencies(blur_test libgputk)

# Set output directories
set_target_properties(blur blur_datagen blur_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")
