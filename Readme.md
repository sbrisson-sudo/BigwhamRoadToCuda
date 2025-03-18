# Simple prototype to test CUDA linear algebra routines on blocks sparse matrices.

This is meant to be later used to improve a hierachical matrix library, where the matrix is compressed as :
- a set of dense blocks
- a set of low ranks approximated blocks

We want to improve the speed of the matrix-vector operation on these matrices, this operation involves multiple smaller matrix-vector operations (hence this prototype).

To compile on a linux machine with intel opeapi and the cuda toolkit installed :
```bash
cmake -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.0/bin/icpx -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 -DBLAS_LIBRARIES="/opt/intel/oneapi/mkl/2025.0/lib/intel64/libmkl_rt.so" ..
```

Then to run the test script (here for a 3 blocks x 3 blocks block-diagonal matrix with block of size 2 x 2) :
```bash
(base) sbrisson@gelpc7:~/codes/hmat-cuda-prototype-min/build$  ./gpu_hmat_matvec 3 2
n = 3, m = 2
Fake hmat initialized
Matvec time CPU (CPU) = 7.337ms
L2 norm res CPU = 5.28394
Matvec time GPU (cuda streams) = 2.192ms
L2 norm res GPU (cuda streams) = 5.28394
Matvec time GPU (BSR) = 0.67ms
L2 norm res GPU (BSR) = 5.28394
Matvec time GPU (batched) = 0.052ms
L2 norm res GPU (batched) = 0
```

The 3rd GPU matvec operation (batched) is faulty, for a small case the result is incorrect, for larger more realistic sistuations it leads to memory faults :
```bash
(base) sbrisson@gelpc7:~/codes/hmat-cuda-prototype-min/build$  ./gpu_hmat_matvec 32768  64
n = 32768, m = 64
Fake hmat initialized
Matvec time CPU (CPU) = 61.252ms
L2 norm res CPU = 1.35816e+09
Matvec time GPU (cuda streams) = 135.709ms
L2 norm res GPU (cuda streams) = 1.35816e+09
Matvec time GPU (BSR) = 9.397ms
L2 norm res GPU (BSR) = 1.35816e+09
CUDA Runtime Error at: /home/sbrisson/codes/hmat-cuda-prototype-min/src/fakeHMatCUDA.hpp:346
an illegal memory access was encountered cudaMemcpy(y_data, d_y_, vector_size_, cudaMemcpyDeviceToHost)
```