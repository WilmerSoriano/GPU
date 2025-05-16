SRC_PATH=src
BUILD_DIR=build

GPUTK_PATH=/usr/local/libgputk
GPUTK_LIB_PATH=$(GPUTK_PATH)/lib
GPUTK_INCLUDE_PATH=$(GPUTK_PATH)/include
NVCC=nvcc
CXX=g++

CUDA_FLAGS=-I $(GPUTK_INCLUDE_PATH) -I /usr/local/cuda/include
CXX_FLAGS=-std=c++11 -I $(GPUTK_INCLUDE_PATH) -I /usr/local/cuda/include
LIBS=-lm -L $(GPUTK_LIB_PATH) -lgputk

# SOURCES := $(wildcard $(SRC_PATH)/*.cpp)
OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
CUDA_SOURCES := $(wildcard $(SRC_PATH)/*.cu)
CUDA_OBJECTS := $(CUDA_SOURCES:%=$(BUILD_DIR)/%.o)
DATAGEN_SOURCES := $(SRC_PATH)/dataset_generator.cpp
DATAGEN_OBJECTS := $(DATAGEN_SOURCES:%=$(BUILD_DIR)/%.o)

TEST_EXE=$(BUILD_DIR)/main_test

# Dataset Generator Main
datagen: $(BUILD_DIR) $(BUILD_DIR)/datagen

$(BUILD_DIR)/datagen: $(DATAGEN_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $(BUILD_DIR)/datagen $^ $(LIBS)

# Testing Main
test: $(BUILD_DIR) $(TEST_EXE)

$(TEST_EXE): $(BUILD_DIR)/main_test.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LIBS)

# Testing object file
$(BUILD_DIR)/main_test.o: $(SRC_PATH)/main_test.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# C++ files
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(CUDA_FLAGS) -g -c $< -o $@

# CUDA files
$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -lineinfo -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/src

clean:
	rm -rf $(BUILD_DIR)

.SUFFIXES: .c .cu .o
.PHONY: all test clean