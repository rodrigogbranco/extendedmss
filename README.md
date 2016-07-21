# extendedmss
Extended MSS (Maximum Subarray Sum) Implementations (Sequential, OpenMP, MPI e CUDA)

- cuda_perumalla_arch_21.cu is used for CUDA GPUs with ARCH=2.1 or less
- cuda_perumalla_arch_30.cu is used for CUDA GPUs with ARCH=3.0 or higher
- You can use gerador1.c to generate the input for implementations with: ./gerador1 N > inputfile, and it will generate a file with 2^N numbers
- mpi_example.sh is an example of a submission job on Cluster CTEI through PBS scheduler. You can use it to run CUDA, OpenMP and Sequential implementations too, changing the compiling line and the call of executable.

We compiled our codes with:
g++ -o gerador1 gerador1.c -fopenmp
g++ -o <filename> seq1d_perumalla.cpp -fopenmp
g++ -o <filename> seq1d_perumalla_openmp.cpp -fopenmp
nvcc -arch=compute_XX -code=sm_XX -o <filename> cuda_perumalla_v4.cu -Xcompiler -fopenmp

PS1: Change compute_XX and sm_XX to your device specifications. We have used compute_20, sm_21 for GT 720M and GTX 460 v2, compute_30, sm_30 fo GTX 680 and compute_35, sm_35 for GTX Titan Black.

PS2: If the CUDA implementation is not working on your machine, check the SDK version. We tested it on SDK 7.5, and for SDK 4.2 we need to:
- Comment #include <thrust/execution_policy.h> directive
- Remove all thrust::host and thrust::device arguments in thrust calls.
