# dev/sycl

This directory contains hand-written SYCL kernels used in the main GPT-2
training SYCL code, [`train_gpt2.sycl.cpp`](../../train_gpt2.sycl.cpp).

## Prerequisites
See [`README.md`](../../README.md) in the root directory of this repository for
instructions on setting up pre-requisites.

## Standalone kernels for experiments
Once the prerequisites are set up, running the standalone kernels placed in
this directory should be easy to build and run. Modify and use the
run script, [`run.sh`](run.sh), to compile and run the
standalone kernels in that directory. Here's how to modify the run script:
* At the top of the file, there are some CPU/GPU devices listed as:
  ```sh
  # CPUs
  ...
  # Intel GPUs
  ...
  # NVidia GPUs
  ...
  ```
  Uncomment one of the sets based on the hardware you are planning to compile
  for. The `aot` variable listed there is optional for Intel devices, but
  apparently required for Nvidia GPUs. Use `aot=a100` if compiling for Nvidia
  A100 GPUs, or use `aot=h100` for compiling for Nvidia H100 GPUs.
* After that are some build options. If you have built the open-source
  [oneMKL Interfaces](https://github.com/oneapi-src/oneMKL/) project to link to
  for additional speed, set `onemkl_interfaces=yes`, and correct the
  `ONEMKL_ROOT` environment variable that immediately follows that to point to
  the oneMKL Interfaces build.
* If you are targeting CPU device and have TBB installed in standard paths,
  then comment out the part where the script sources TBB environment from a
  non-standard path.
* Lastly, set up the path to the compiler; if it is installed in a standard
  path, then you can comment out the part where the script sources the compiler
  environment. Otherwise, (e.g., if you have built DPC++-LLVM on Nvidia from
  scratch, for instance,) set up the correct `$PATH`/`$LD_LIBRARY_PATH`
  variables to point to the compiler.

The last few lines of the script have the `make ... run_all` command that
both builds and runs all the standalone kernels, which should take some
time to run since there are many kernels and that are repeatedly run to
benchmark their performance. You can build and run individual files by
replacing `run_all` with the file name without the `.sycl.cpp` extension.
For example, for building and running only `matmul_forward.sycl.cpp`,
replace `run_all` with `run_matmul_forward`. You should see an output like:

```sh
$ ./run.sh
clang version 19.0.0git (https://github.com/intel/llvm.git 44c34c14326a189f719fbbe3393a8ee4a790f1c2)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /localdisk2/mkl/gchoudha/cache/stash/llvm-oss-compiler/build-2024-08-17/lnx/cuda/compiler/bin
Build config: +assertions
All available SYCL devices:
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA H100 PCIe 9.0 [CUDA 12.6]
<------------- Truncated ------------->
clang++ -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -DLLMSYCL -march=native -ffp-model=precise -DSYCL_CUDA -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90 -D_DISABLE_SG_SIZE_8 -D_DISABLE_SG_SIZE_16   matmul_forward.sycl.cpp -lsycl -lOpenCL -o matmul_forward
========================================
Running matmul_forward...
========================================
./matmul_forward
Device maximum workgroup size = 1024
Device sub_group sizes = [32, ]
Running reference CPU kernel.
kernel ref OMP | time 71455.3197 ms | tflops 0.00
************************************************.
Checking kernel set #1.
<------------- Truncated ------------->
All results match. Benchmarking kernel 1.
kernel  1 | time 416.1700 ms | tflops 0.37

************************************************.
Checking kernel set #2.
<------------- Truncated ------------->
All results match. Benchmarking kernel 2.
kernel  2 | time 110.0388 ms | tflops 1.41

************************************************.
Checking kernel set #3.
Testing sg_size = 32
Checking kernel 3<1, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 3<1, 32>.
kernel  3< 1, 32> | time 76.5156 ms | tflops 2.02
Checking kernel 3<2, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 3<2, 32>.
kernel  3< 2, 32> | time 44.1589 ms | tflops 3.50
Checking kernel 3<4, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 3<4, 32>.
kernel  3< 4, 32> | time 40.9060 ms | tflops 3.78
Checking kernel 3<8, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 3<8, 32>.
kernel  3< 8, 32> | time 40.5502 ms | tflops 3.81
Checking kernel 3<16, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 3<16, 32>.
kernel  3<16, 32> | time 41.6397 ms | tflops 3.71
Checking kernel 3<32, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 3<32, 32>.
kernel  3<32, 32> | time 42.2843 ms | tflops 3.66


************************************************.
Checking kernel set #4.
Testing sg_size = 32
Checking kernel 4<1, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 4<1, 32>.
kernel  4< 1, 32> | time 399.2595 ms | tflops 0.39
Checking kernel 4<2, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 4<2, 32>.
kernel  4< 2, 32> | time 399.2614 ms | tflops 0.39
Checking kernel 4<4, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 4<4, 32>.
kernel  4< 4, 32> | time 399.2663 ms | tflops 0.39
Checking kernel 4<8, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 4<8, 32>.
kernel  4< 8, 32> | time 399.2618 ms | tflops 0.39
Checking kernel 4<16, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 4<16, 32>.
kernel  4<16, 32> | time 399.2625 ms | tflops 0.39
Checking kernel 4<32, 32>.
<------------- Truncated ------------->
All results match. Benchmarking kernel 4<32, 32>.
kernel  4<32, 32> | time 399.2624 ms | tflops 0.39
```

## License

[MIT](../../LICENSE).
