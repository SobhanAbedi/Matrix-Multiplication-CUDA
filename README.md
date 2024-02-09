# CUDA Matrix Multiplication
In this program multiple implementations of parallel matrix multiplication have been realized using CUDA kernels and Their computation time can be easily compared for a better understanding of GP-GPU (an SIMD processor) architecture.

## Kernels
The following steps have been taken in order to improve the multiplication process.
1. Simple parallel matrix multiplication
2. Fix the output matrix size limitation
	1. Naive fix
	2. Using Blocks
	3. Tiling
3. Speeding up the process using ***Shared Memory***

### matrixMulCUDA_1:
This kernel allocates one output cell to each thread and it simply does the MAC computations for that single cell.
It is only meant to be run with a single block and that limits the output matrix size due to thread limitations in each computation block.
```
Process time:
32^3: 0.1(msec)
```

### matrixMulCUDA_2_1:
Similar to first kernel, in This kernel each thread calculates the results for a unified n x n block of the output matrix. If n is set to 1, This would become the same as the previous method. In order to remove size limitation, n is set dynamically in this kernel.
But This method is very slow due to bad use of SIMD structure.
```
Process time:
32^3:	0.1(msec)
512^3:	72.9(msec)
4096^3:	54,551.7(msec)
```

### matrixMulCUDA_2_2:
Similar to first kernel, in this kernel each thread calculates a single output cell but multiple blocks of threads are used in order to make the program scalable with output matrix size.
This method is much faster than 2.1 but due to inherent limitation of GPUs, the number of blocks that can be schedule in one kernel call is limited and therefor maximum matrix size will be limited.
```
Process time:
512^3:	2.0(msec)
8192^3:	7,451.3(msec)
```

### matrixMulCUDA_2_3:
In order to fully remove output matrix size limitation and speedup the process, The previous two methods have been combined. This implementation not as fast as 2.2 but also doesn't have the output matrix size limitation.
It still suffers from inefficient use of SIMD structure inside each thread. This issue will be fixed in the next kernel.
```
Process time:
512^3:	5.7(msec)
8192^3:	34,531.6(msec)
```

### matrixMulCUDA_3:
This method is similar to 2.3 but two changes have been made:
1. Each thread now calculates a mesh of output cells instead of a unified block. This is the correct way to use an SIMD structure.
2. Small blocks of input matrices are kept in Shared memory. Due to matrix multiplication having a process time of O(n^3), input values are used multiple times and in order to take advantage of this data reuse, it is critical to hold that pieces of data in SMEM.

This method is also not fully scalable with output matrix size. Because Tiles are placed inside Shared Memory and due to it's limited size, Tiles cannot grow beyond 64x64. This means that for each tile of 64x64 we need a separate block and that introduces two limitations:
1. Maximum matrix size will be limited. But it will be a very large limit and due to O(n^3) nature of this process, at that point this solution would not be a good answer to that problem anymore and a distributed solution will be needed.
2. Data reuse will be limited to a block of 64x64 which will result in Flops/Byte ratio of (2 * 64^3) / (3 * 4 * 64^2) = 10.67
```
Process time:
512^3:	2.3(msec)
8192^3:	5,504.1(msec)
```

## Further Work
All of the methods implemented are using the basis of inner multiplication. Using outer multiplication could result in higher data reuse and could potentially make the solution fully scalable with matrix size.