# extendedmss
Extended MSS (Maximum Subarray Sum) Implementations (Sequential, OpenMP, MPI e CUDA)

- cuda_perumalla_arch_21.cu is used for CUDA GPUs with ARCH=2.1 or less
- cuda_perumalla_arch_30.cu is used for CUDA GPUs with ARCH=3.0 or higher
- You can use gerador1.c to generate the input for implementations with: ./gerador1 N > inputfile, and it will generate a file with 2^N numbers
- mpi_example.sh is an example of a submission job on Cluster CTEI through PBS scheduler. You can use it to run CUDA, OpenMP and Sequential implementations too, changing the compiling line and the call of executable.
