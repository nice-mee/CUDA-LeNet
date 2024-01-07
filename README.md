# CUDA-LeNet
Course final project for ECE 408 FA23 @ UIUC. For original project description, see [_README.md](_README.md).

## Optimization Techniques
For a more comprehensive discussion, see [report.pdf](report.pdf).

### 1. Transforming Convolution to Matrix Multiplication (im2col)
This optimization works by transforming the image from `B * C * H * W` to `B * (C * K * K) * (H_out * W_out)`, where `H_out` and `W_out` are the output height and width, respectively. This transformation is done in a kernel called `unroll_kernel` in [new-forward_opt1_matrix.cu](opt_archive/new-forward_opt1_matrix.cu). The mask can preserve its original shape, from `M * C * K * K` to `M * (C * K * K)`. The magic here is that this requires no changes to the mask at all due to the way we store it in memory. Now the computation becomes performing a `M * (C * K * K) * (H_out * W_out)` matrix multiplication, which is done in [new-forward_opt1_matrix.cu](opt_archive/new-forward_opt1_matrix.cu) in the kernel `mat_forward_kernel`.

### 2. Kernel Fusion for Unrolling and Matrix Multiplication
Fusing kernels is a common practice in CUDA optimization. Instead of launching two kernels and constantly reading and writing in the first kernel then reading and writing global memory in the second kernel again, we can fuse all the operations into one kernel and eliminate all the internal global memory accesses.

### 3. Using Tensor Cores to Speed Up Matrix Multiplication
This optimization works because nVidia designed some dedicated hardware for matrix multiplication, the Tensor Cores. These cores compute much faster than traditional CUDA matmul kernels in fixed-size matrix multiplications.

### 4. Index Reusing in Fusion Kernel
The reason why this trick works is explained in the report. But the basic idea is that in this case the kernel is not compute-bounded by floating point operations, but rather integer operations used to calculate the indices (not common, but still possible). So reusing the indices can dramatically improve the performance.

### 5. Multiple Kernel Implementations for Different Layer Sizes
This optimization is dependent on the fact that Layer 1 has a small `M` size (`M=4`), this is much smaller than the size used by Tensor Cores (16\*16\*16). Luckily, Tensor Cores also support a smaller size in fist dimension: 8\*32\*16. So we can use this size for Layer 1 and use the original size for other layers.

## Performance Results

| Implementation | Batch Size | Op Time 1 | Op Time 2 |
| -------------- | ---------- | --------- | --------- |
| Baseline       | 100        | 0.18ms    | 0.65ms    |
| Opt 1          | 100        | 1.45ms    | 1.26ms    |
| Opt 2          | 100        | 0.63ms    | 0.35ms    |
| Opt 3          | 100        | 0.58ms    | 0.30ms    |
| Opt 4          | 100        | 0.21ms    | 0.14ms    |
| Opt 5          | 100        | 0.16ms    | 0.15ms    |

| Implementation | Batch Size | Op Time 1 | Op Time 2 |
| -------------- | ---------- | --------- | --------- |
| Baseline       | 1000       | 1.71ms    | 6.39ms    |
| Opt 1          | 1000       | 13.69ms   | 10.87ms   |
| Opt 2          | 1000       | 6.21ms    | 3.28ms    |
| Opt 3          | 1000       | 5.71ms    | 2.91ms    |
| Opt 4          | 1000       | 2.00ms    | 1.24ms    |
| Opt 5          | 1000       | 1.42ms    | 1.25ms    |

| Implementation | Batch Size | Op Time 1 | Op Time 2 |
| -------------- | ---------- | --------- | --------- |
| Baseline       | 5000       | 8.49ms    | 32.13ms   |
| Opt 1          | 5000       | 67.65ms   | 55.03ms   |
| Opt 2          | 5000       | 30.98ms   | 16.44ms   |
| Opt 3          | 5000       | 28.50ms   | 14.55ms   |
| Opt 4          | 5000       | 10.04ms   | 6.16ms    |
| Opt 5          | 5000       | 7.01ms    | 6.16ms    |

The final competition is run on `B=10000`, this submission is the fastest among all the submissions in the class, with a total op time of 25.85ms.
