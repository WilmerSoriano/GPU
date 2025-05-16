set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")

# Create main executable
add_executable(grad_image ${CUDA_SOURCES} "${SRC_PATH}/grad_image.cpp" "${SRC_PATH}/utils.cpp")
target_include_directories(grad_image PRIVATE ${COMMON_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_directories(grad_image PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(grad_image ${COMMON_LIBS} ${OpenCV_LIBS})
add_dependencies(grad_image libgputk)

# Set output directories
set_target_properties(grad_image
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")