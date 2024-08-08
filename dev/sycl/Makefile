# Makefile for building dev/sycl kernels
# Collects all the make commands in one file but each file also
# has the compile and run commands in the header comments section.

ifeq ($(DEVICE),cpu)
    NUMACTL_CMD=numactl -i all
endif

# Find icpx (Intel DPC++/SYCL compiler) or open source LLVM clang++ compiler
SYCLCC := $(shell which icpx 2>/dev/null)
ifeq ($(notdir $(SYCLCC)),icpx)
    $(info icpx compiler found at $(SYCLCC))
else
    $(info icpx not found. Searching for clang++.)
    SYCLCC := $(shell which clang++ 2>/dev/null)
    ifeq ($(SYCLCC),)
        $(error clang++ not found either.)
    else
        $(info clang++ found at: $(SYCLCC))
    endif
endif

# SYCL flags
ifneq ($(DEBUG),yes)
    SYCL_FLAGS = -O3
else
    SYCL_FLAGS = -g -O0
    EMU=gdb
endif
SYCL_FLAGS += -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -DLLMSYCL
SYCL_LDFLAGS =
SYCL_INCLUDES =
SYCL_LDLIBS = -lsycl -lOpenCL
ifeq ($(notdir $(SYCLCC)),icpx)
    SYCL_FLAGS += -xhost -qopenmp -qopenmp-simd -DOMP -fp-model=precise
    SYCL_LDLIBS = -liomp5
else ifeq ($(notdir $(SYCLCC)),clang++)
    SYCL_FLAGS += -march=native -ffp-model=precise
    # Unfortunately clang doesn't seem to have OpenMP out-of-the-box
    # and must be built/installed manually AFAIK
    #SYCL_FLAGS += -fopenmp=libomp -fopenmp-simd -DOMP
endif

ifneq ($(TIME_PROFILE),)
    SYCL_FLAGS += -DTIMEPROFILE=$(TIME_PROFILE)
endif
ifneq ($(DEVICE),gpu)
    SYCL_LDLIBS += -ltbb
endif

# Ahead-of-time compilation flags, if possible
ifeq ($(SYCL_AOT_COMPILE),pvc)
    SYCL_FLAGS += -fsycl-targets=spir64_gen -Xs "-device pvc"
    SYCL_FLAGS += -D_DISABLE_SG_SIZE_8
endif
ifeq ($(SYCL_AOT_COMPILE),a100)
    SYCL_FLAGS += -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80
    SYCL_FLAGS += -D_DISABLE_SG_SIZE_8 -D_DISABLE_SG_SIZE_16
endif
ifeq ($(SYCL_AOT_COMPILE),h100)
    SYCL_FLAGS += -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90
    SYCL_FLAGS += -D_DISABLE_SG_SIZE_8 -D_DISABLE_SG_SIZE_16
endif

# Default rule for our SYCL files
%: %.sycl.cpp
	$(SYCLCC) $(SYCL_FLAGS) $(SYCL_INCLUDES) $(SYCL_LDFLAGS) $^ $(SYCL_LDLIBS) -o $@

# Build all targets
TARGETS = matmul_forward matmul_backward \
          layernorm_forward layernorm_backward \
          attention_forward attention_backward \
          encoder_forward encoder_backward \
          gelu_forward gelu_backward \
          softmax_forward crossentropy_forward crossentropy_softmax_backward

all: $(TARGETS)

# Run all targets
run_all: all
	@for target in $(TARGETS); do \
		echo "========================================"; \
		echo "Running $$target ..."; \
		echo "========================================"; \
		$(NUMACTL_CMD) ./$$target; \
	done

# Run specific target, e.g., "run_matmul_forward"
run_%: %
	@echo "========================================"
	@echo "Running $^..."
	@echo "========================================"
	$(NUMACTL_CMD) $(EMU) ./$^

# Clean up
clean:
	rm -f $(TARGETS)
