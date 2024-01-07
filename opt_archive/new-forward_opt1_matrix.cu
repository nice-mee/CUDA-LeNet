#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void mat_forward_kernel(const float *A, const float *B, float *C, int numARows, int numAColumns, int numBColumns)
{
  int Batch = blockIdx.z;
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  float Cvalue = 0.0;

  for (int i = 0; i < ceil(numAColumns/(float) TILE_WIDTH); i++) {
    if (Row < numARows && i * TILE_WIDTH + threadIdx.x < numAColumns)
      tileA[threadIdx.y][threadIdx.x] = A[Row * numAColumns + (i * TILE_WIDTH + threadIdx.x)];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0;
    if (Col < numBColumns && i * TILE_WIDTH + threadIdx.y < numAColumns)
      tileB[threadIdx.y][threadIdx.x] = B[Batch * (numAColumns * numBColumns) + Col + (i * TILE_WIDTH + threadIdx.y) * numBColumns];
    else
      tileB[threadIdx.y][threadIdx.x] = 0.0;
    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; j++) {
      Cvalue += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (Row < numARows && Col < numBColumns)
    C[Batch * (numARows * numBColumns) + Row * numBColumns + Col] = Cvalue;
}

__global__ void unroll_kernel(const float *X, float *X_unroll, const int C , const int K, const int H, const int W, const int S, const int W_size)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int w_base = c * K * K;
    const int h = (blockIdx.z / W_size) * TILE_WIDTH + threadIdx.y;
    const int w = (blockIdx.z % W_size) * TILE_WIDTH + threadIdx.x;

    
    #define X_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define X_unroll_3d(i2, i1, i0) X_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]

    if (h < H_out && w < W_out) {
      for (int p = 0; p < K; p++) {
          for (int q = 0; q < K; q++) {
              const int h_unroll = w_base + p * K + q;
              const int w_unroll = h * W_out + w;
              X_unroll_3d(b, h_unroll, w_unroll) = X_4d(b, c, h * S + p, w * S + q);
          }
      }
    }

    #undef X_4d
    #undef X_unroll_3d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMalloc((void **)device_input_ptr, B*C*H*W*sizeof(float));
    cudaMalloc((void **)device_output_ptr, B*M*H_out*W_out*sizeof(float));
    cudaMalloc((void **)device_mask_ptr, M*C*K*K*sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int W_size = ceil((float)W_out/TILE_WIDTH);
    int H_size = ceil((float)H_out/TILE_WIDTH);

    float *device_unroll;
    cudaMalloc((void **)&device_unroll, B*C*K*K*H_out*W_out*sizeof(float));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, C, (W_size * H_size));
    unroll_kernel<<<dimGrid, dimBlock>>>(device_input, device_unroll, C, K, H, W, S, W_size);

    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    dimGrid = dim3(ceil((H_out*W_out)/(float)TILE_WIDTH), ceil(M/(float)TILE_WIDTH), B);
    mat_forward_kernel<<<dimGrid, dimBlock>>>(device_mask, device_unroll, device_output, M, C*K*K, H_out*W_out);

    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    cudaFree(device_unroll);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, B*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
