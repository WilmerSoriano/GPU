set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")
file(GLOB DATAGEN_SOURCES "${SRC_PATH}/dataset_generator.cpp" "${SRC_PATH}/color_to_grayscale_cpu.cpp")

# Create main executable
add_executable(rgb2gray ${CUDA_SOURCES} "${SRC_PATH}/main.cpp")
target_include_directories(rgb2gray PRIVATE 
    ${COMMON_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(rgb2gray PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(rgb2gray
    ${COMMON_LIBS}
    ${OpenCV_LIBS}
)
add_dependencies(rgb2gray libgputk)

# Create datagen executable
add_executable(rgb2gray_datagen ${DATAGEN_SOURCES})
target_include_directories(rgb2gray_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(rgb2gray_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(rgb2gray_datagen ${COMMON_LIBS})
add_dependencies(rgb2gray_datagen libgputk)

# Create test executable
add_executable(rgb2gray_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp")
target_include_directories(rgb2gray_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(rgb2gray_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(rgb2gray_test ${COMMON_LIBS})
add_dependencies(rgb2gray_test libgputk)

# Set output directories
set_target_properties(rgb2gray rgb2gray_datagen rgb2gray_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")