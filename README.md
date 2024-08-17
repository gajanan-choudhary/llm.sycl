# llm.sycl

This is a SYCL port of [karpathy/llm.c](https://github.com/karpathy/llm.c). The
goal is to have at least the FP32 version work on x86 CPUs, Intel GPUs, and
Nvidia GPUs. This repository has only been tested on Linux.

## Prerequisites and notes

In all cases, the required OS is Linux which should have GNU Make pre-installed.
Optionally, for additional speed relying on vendor-specific libraries (Intel
oneMKL, cuBLAS, etc.), you can build the open-source
[oneMKL Interfaces](https://github.com/oneapi-src/oneMKL/) project, which should
contain the same prerequisites listed below, and link to it when building the
SYCL code in this repository.

### x86 CPUs
* oneAPI Base toolkit containing:
    * Latest Intel DPC++ Compiler (icpx/icx)
    * TBB library
* For standalone kernels, OpenMP (which comes with the compiler) helps improve
  the runtime for the reference solution calculation for validation.

### Intel GPUs
* oneAPI Base toolkit containing:
    * Latest Intel DPC++ Compiler (icpx/icx)
* Be a member of `render` and `video` Linux groups
    * This can be done by running `sudo usermod -a -G video $USER` and
      `sudo usermod -a -G render $USER`.
* For standalone kernels, OpenMP (which comes with the compiler) helps improve
  the runtime for the reference solution calculation for validation.

### Nvidia GPUs
This is the hardest part of the entire repository, have patience!
* Latest CUDA toolkit (tested on CUDA 12.6)
* Latest builds of the open-source
  [DPC++ LLVM compiler](https://github.com/intel/llvm/releases/)
  (tested on [nightly-2024-08-07](https://github.com/intel/llvm/releases/tag/nightly-2024-08-07)
  build) containing the CLang++-LLVM (`clang++`).
    * In the worst-case, follow
      "[Compiling SYCL for Different GPUs](https://www.intel.com/content/www/us/en/developer/articles/technical/compiling-sycl-with-different-gpus.html)"
      page to simply build the DPC++-LLVM compiler directly from scratch
      (should take 10 minutes). Following that, you should be able to run
      `sycl-ls` and see your Nvidia GPU listed, e.g.:
      ```sh
      [cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA H100 PCIe 9.0 [CUDA 12.6]
      ```
      This method was also tested.
    * Note: It is possible that following Codeplay's instructions to use the Intel
      DPC++ compiler with some additional steps also works, see
      [oneAPI for Nvidia GPUs](https://developer.codeplay.com/products/oneapi/nvidia/latest/guides/get-started-guide-nvidia),
      but this has neither been tested, nor are the Makefiles set up that way.
* Be a member of `render` and `video` Linux groups
    * This can be done by running `sudo usermod -a -G video $USER` and
      `sudo usermod -a -G render $USER`.
* For standalone kernels, OpenMP (which comes with the compiler) helps improve
  the runtime for the reference solution calculation for validation. However,
  the open-source LLVM compiler (`clang`/`clang++`) does not automatically
  install OpenMP out-of-the-box. This must be done manually, which is left to
  users to figure out. In the worst case, standalone tests would run a bit
  slower.

## Quick start

Follow
[karpathy/llm.c/blob/master/README.md#quick-start-cpu](https://github.com/karpathy/llm.c/blob/master/README.md#quick-start-cpu)
to set up the training/testing datasets once by running:
```sh
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
```

Once the prerequisites are set up, the rest should be easy. Modify and use the
run script, [`run.sh`](run.sh), to compile the GPT-2 FP32 SYCL code,
[`train_gpt2.sycl.cpp`](train_gpt2.sycl.cpp), into a binary and run it. Here's
how to modify the run script:
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
* The last few lines of the script are to build and run
  [`train_gpt2.sycl.cpp`](train_gpt2.sycl.cpp) and
  [`test_gpt2.sycl.cpp`](test_gpt2.sycl.cpp) to run both training and
  testing. Modify those based on what you want to achieve.

That's it. Running the script should give you an output like:

```sh
clang version 19.0.0git (https://github.com/intel/llvm.git 44c34c14326a189f719fbbe3393a8ee4a790f1c2)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /localdisk2/mkl/gchoudha/cache/stash/llvm-oss-compiler/build-2024-08-17/lnx/cuda/compiler/bin
Build config: +assertions
All available SYCL devices:
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA H100 PCIe 9.0 [CUDA 12.6]
<------------- Truncated ------------->
clang++ -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -DLLMSYCL -march=native -ffp-model=precise -DTIMEPROFILE=1 -DSYCL_CUDA -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90   train_gpt2.sycl.cpp -lsycl -lOpenCL -o train_gpt2sycl
clang++ -O3 -fsycl -fno-sycl-id-queries-fit-in-int -std=c++17 -DLLMSYCL -march=native -ffp-model=precise -DTIMEPROFILE=1 -DSYCL_CUDA -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90   test_gpt2.sycl.cpp -lsycl -lOpenCL -o test_gpt2sycl
Submit training run: train_gp2sycl
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73347840
val loss 5.325533
step    0: train loss 4.677785 (took     79.346 ms: [     0.008,     19.071,      0.009,     57.550,      2.707])
step    1: train loss 5.191640 (took     77.752 ms: [     0.010,     18.821,      0.409,     56.571,      1.941])
step    2: train loss 4.438737 (took     77.963 ms: [     0.007,     19.029,      0.464,     56.519,      1.944])
<------------- Truncated ------------->
step   19: train loss 4.552019 (took     77.882 ms: [     0.005,     18.797,      0.401,     56.736,      1.943])
val loss 4.329034
generating:
---

I am Franklin in this world.
My brother and father drow an ear off the prince.
Our enemy, my brother and father,
Were the great Duke of the same age?

<|endoftext|>PUMAS:
And then we are in peace, shrine on shrine.
As close as cattle can
---
step   20: train loss 4.527010 (took     78.513 ms: [     0.006,     18.964,      0.401,     57.201,      1.941])
<------------- Truncated ------------->
```

## Standalone kernels for experiments
* The [`dev/sycl`](dev/sycl) directory contains many standalone kernels and
  a [`run.sh`](dev/sycl/run.sh) script similar to the one in the root directory,
  that can be modified and run similarly as explained above. At the end of the
  script is the `make ... run_all` command that both builds and runs all tests,
  which should take some time. Target individual tests by replacing `run_all`
  with the file name without the `.sycl.cpp` extension. For example, for
  building and running only `matmul_forward.sycl.cpp`, replace `run_all` with
  `run_matmul_forward`. See [`dev/sycl/README.md`](dev/sycl/README.md) for
  sample output.

## License

[MIT](LICENSE).
