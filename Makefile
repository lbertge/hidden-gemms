# Compiler settings
NVCC = nvcc
CFLAGS = -O3  # Adjust sm_XX based on your GPU architecture
CUDA_PATH = /usr/local/cuda  # Adjust if your CUDA installation is elsewhere

# Directories
SRC_DIR = kernels
PROFILING_DIR = profiling
BUILD_DIR = build
BIN_DIR = bin

# Find all .cu files in the source and profiling directories
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
PROFILING_SOURCES = $(wildcard $(PROFILING_DIR)/*.cu)

# Create target executable names from source files
EXECUTABLES = $(SOURCES:$(SRC_DIR)/%.cu=$(BIN_DIR)/%)
PROFILING_EXECUTABLES = $(PROFILING_SOURCES:$(PROFILING_DIR)/%.cu=$(BIN_DIR)/%)

# Add cuBLAS library flags
LIBS = -lcublas

# Default target
all: setup $(EXECUTABLES) $(PROFILING_EXECUTABLES)

# Create build and bin directories
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Compile rule for regular .cu files
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(CFLAGS) $< $(LIBS) -o $@

# Compile rule for profiling .cu files
$(BIN_DIR)/%: $(PROFILING_DIR)/%.cu
	$(NVCC) $(CFLAGS) $< $(LIBS) -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(BIN_DIR)/*

# Print available targets
list:
	@echo "Available targets:"
	@echo $(EXECUTABLES) $(PROFILING_EXECUTABLES) | tr ' ' '\n' | sed 's/^/- /'

# Add device_info to utilities
device_info: utils/device_info.cu
	$(NVCC) $(NVCC_FLAGS) -o bin/device_info utils/device_info.cu $(LIBS)

# Add device_info to the all target
all: gemm device_info

.PHONY: all setup clean list 