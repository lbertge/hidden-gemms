# Compiler settings
NVCC = nvcc
# CFLAGS = -O3 -arch=sm_75  # Adjust sm_XX based on your GPU architecture
CUDA_PATH = /usr/local/cuda  # Adjust if your CUDA installation is elsewhere

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Find all .cu files in the current directory
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
# Create target executable names from source files
EXECUTABLES = $(SOURCES:$(SRC_DIR)/%.cu=$(BIN_DIR)/%)

# Default target
all: setup $(EXECUTABLES)

# Create build and bin directories
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Compile rule for .cu files
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Print available targets
list:
	@echo "Available targets:"
	@echo $(EXECUTABLES) | tr ' ' '\n' | sed 's/^/- /'

.PHONY: all setup clean list 