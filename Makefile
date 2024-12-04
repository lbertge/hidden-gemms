NVCC = nvcc
CUDA_PATH = /usr/local/cuda  # Adjust based on your system

# Directories
SRC_DIR = src
BENCH_DIR = benchmark
PROFILING_DIR = profiling
BIN_DIR = bin

# Find all .cu files in the source and profiling directories
SRC = $(wildcard $(SRC_DIR)/*.cu)
BENCHS = $(wildcard $(BENCH_DIR)/*.cu)
PROFILINGS = $(wildcard $(PROFILING_DIR)/*.cu)
UTILS = $(wildcard $(SRC_DIR)/utils/*.cu)

# Create target executable names from source files
BENCH_EXECUTABLES = $(BENCHS:$(BENCH_DIR)/%.cu=$(BIN_DIR)/%)
PROFILING_EXECUTABLES = $(PROFILINGS:$(PROFILING_DIR)/%.cu=$(BIN_DIR)/%)
UTILS_EXECUTABLES = $(UTILS:$(SRC_DIR)/utils/%.cu=$(BIN_DIR)/%)

# Compiler flags
2070 = -gencode arch=compute_75,code=sm_75
4090 = -gencode arch=compute_86,code=sm_86
ARCH = $(4090)

# CFLAGS = $(ARCH) -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17
CFLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17

# Library flags (e.g., cuBLAS)
LIBS = -lcublas

# Default target
all: setup $(BENCH_EXECUTABLES) $(PROFILING_EXECUTABLES) $(UTILS_EXECUTABLES)

# Create build and bin directories
setup:
	@mkdir -p $(BIN_DIR)

# Compile rule for regular .cu files
$(BIN_DIR)/%: $(BENCH_DIR)/%.cu $(SRC)
	$(NVCC) $(CFLAGS) $< $(SRC) $(LIBS) -o $@

# Compile rule for profiling .cu files
$(BIN_DIR)/%: $(PROFILING_DIR)/%.cu $(SRC)
	$(NVCC) $(CFLAGS) $< $(SRC) $(LIBS) -o $@

# Print device information
$(BIN_DIR)/%: $(UTILS)/%.cu
	$(NVCC) $(CFLAGS) $< $(LIBS) -o $@

# Clean build files
clean:
	rm -f $(BIN_DIR)/*

# Print available targets
list:
	@echo "Available targets:"
	@echo $(BENCH_EXECUTABLES) $(PROFILING_EXECUTABLES) | tr ' ' '\n' | sed 's/^/- /'

.PHONY: all setup clean list device_info