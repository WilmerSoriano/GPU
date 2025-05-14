SRC_PATH=src
BUILD_DIR=build

LIB_PATH=$(CURDIR)/lib
GPUTK_LIB_PATH=$(LIB_PATH)/libgputk/lib
GPUTK_SRC_PATH=$(LIB_PATH)/libgputk
NVCC=nvcc
CXX=g++

CUDA_FLAGS=-I $(GPUTK_SRC_PATH)
CXX_FLAGS=-std=c++11 -I $(GPUTK_SRC_PATH)
LIBS=-lm -L $(GPUTK_LIB_PATH) -lgputk

SOURCES := $(wildcard $(SRC_PATH)/*.cu)
OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
DATAGEN_SOURCES := $(wildcard $(SRC_PATH)/*.cpp)
DATAGEN_OBJECTS := $(DATAGEN_SOURCES:%=$(BUILD_DIR)/%.o)

.PHONY: all
.SUFFIXES: .c .cu .o

all: $(BUILD_DIR) $(BUILD_DIR)/main

$(BUILD_DIR)/main: $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

datagen: $(DATAGEN_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $(BUILD_DIR)/datagen $^ $(LIBS)

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/src

clean:
	rm -rf $(BUILD_DIR)
