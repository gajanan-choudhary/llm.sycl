#!/bin/bash

local_cache=/export/users/${USER}/cache/stash

# CPUs
#vendor=intel
#device=cpu
#backend=opencl

# Intel GPUs
#vendor=intel
#device=gpu
#backend=level_zero
#aot=pvc # Optional ahead-of-time (AOT) compilation on Intel GPUs

# NVidia GPUs
vendor=nvidia
device=gpu
backend=cuda
aot=a100 #h100 # Compulsory AOT compilation for Nvidia GPUs

# Build options, if any
timeprofile=1 # or 2
debug=no
onemkl_interfaces=no
if [[ "$onemkl_interfaces" == "yes" ]]; then
    export ONEMKL_ROOT=/export/users/$USER/oneMKL-interfaces/build/oneMKL-interfaces
    export LD_LIBRARY_PATH=${ONEMKL_ROOT}/lib:${LD_LIBRARY_PATH}
    export MKLROOT=/export/users/$USER/mkl-build-tests/releases/__mkl2024u2RC_20240605_cev/__release_lnx
    export LD_LIBRARY_PATH=${MKLROOT}/lib:${LD_LIBRARY_PATH}
fi

if [[ "$device" == "cpu" || "$onemkl_interfaces" == "yes" ]]; then
    # TBB required for SYCL CPU device
    tbb_build=2021.12.0
    source $local_cache/tbb/${tbb_build}/lnx/package/env/vars.sh
fi
export OMP_NUM_THREADS=112
# NUMA control-related, applicable for CPU devices + OpenMP
#export KMP_AFFINITY=compact,granularity=fine    # If hyperthreading is off
export KMP_AFFINITY=granularity=fine,compact,1,0 # If hyperthreading is on

## Source compiler
# Intel SYCL/DPC++ icpx compiler if running on Intel GPUs
#compiler_build=20240630_nightly
#compiler_build=20240604_rls
#source $local_cache/oneapi-compiler/$compiler_build/lnx/package/setvars.sh
#icpx -fsycl --version
# Open-source LLVM clang++ compiler if running on NVidia GPUs
compiler_build=nightly-2024-08-07
source $local_cache/llvm-oss-compiler/$compiler_build/lnx/cuda/llvm-setvars.sh
clang++ -fsycl --version

echo "All available SYCL devices:"
sycl-ls
export SYCL_DEVICE_ALLOWLIST=
export ONEAPI_DEVICE_SELECTOR="$backend:$device"
echo "Selected device:"
sycl-ls

echo "Cleaning up past builds"
make clean
echo "Beginning new builds"
make -j 2 DEVICE=${device} DEVICE_VENDOR=${vendor} TIME_PROFILE=${timeprofile} DEBUG=${debug} \
          SYCL_AOT_COMPILE=${aot} BUILD_ONEMKL_INTERFACES=${onemkl_interfaces} \
          train_gpt2sycl test_gpt2sycl

echo "Submit training run: train_gp2sycl"
OMP_NUM_THREADS=$omp_num_threads $numa_settings ./train_gpt2sycl
echo "Submit testing run: test_gp2sycl"
OMP_NUM_THREADS=$omp_num_threads $numa_settings ./test_gpt2sycl
