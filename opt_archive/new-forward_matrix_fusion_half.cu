#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16

__global__ void fp32_to_fp16(half *dst, const float *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void fp16_to_fp32(float *dst, const half *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

__global__ void mat_forward_kernel(half *output, const half *input, const half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int b = blockIdx.z;

    __shared__ half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ half tileB[TILE_WIDTH][TILE_WIDTH];

    int numAColumns = C*K*K;
    int numBColumns = H_out*W_out;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    half Cvalue = 0.0;

    for (int i = 0; i < ceil(numAColumns/(float) TILE_WIDTH); i++) {
        int ColA = i * TILE_WIDTH + threadIdx.x;
        int RowB = i * TILE_WIDTH + threadIdx.y;
        int c = RowB / (K*K);
        int p = RowB % (K*K) / K;
        int q = RowB % (K*K) % K;
        int h = Col / W_out;
        int w = Col % W_out;
        if (Row < M && ColA < numAColumns)
            tileA[threadIdx.y][threadIdx.x] = mask[Row * numAColumns + ColA];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        if (Col < numBColumns && RowB < numAColumns)
            tileB[threadIdx.y][threadIdx.x] = in_4d(b, c, h * S + p, w * S + q);
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            Cvalue = __hadd(Cvalue, __hmul(tileA[threadIdx.y][j], tileB[j][threadIdx.x]));
        }
        __syncthreads();
    }
    if (Row < M && Col < numBColumns)
        output[b * (M * numBColumns) + Row * numBColumns + Col] = Cvalue;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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

    
    half *device_input_fp16;
    half *device_mask_fp16;
    half *device_output_fp16;
    cudaMalloc((void **)&device_input_fp16, B*C*H*W*sizeof(half));
    cudaMalloc((void **)&device_mask_fp16, M*C*K*K*sizeof(half));
    cudaMalloc((void **)&device_output_fp16, B*M*H_out*W_out*sizeof(half));

    fp32_to_fp16<<<ceil((float)B*C*H*W/(float)256), 256>>>(device_input_fp16, device_input, B*C*H*W);
    fp32_to_fp16<<<ceil((float)M*C*K*K/(float)256), 256>>>(device_mask_fp16, device_mask, M*C*K*K);

    dim3 dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid = dim3(ceil((H_out*W_out)/(float)TILE_WIDTH), ceil(M/(float)TILE_WIDTH), B);
    mat_forward_kernel<<<dimGrid, dimBlock>>>(device_output_fp16, device_input_fp16, device_mask_fp16, B, M, C, H, W, K, S);

    fp16_to_fp32<<<ceil((float)B*M*H_out*W_out/(float)256), 256>>>(device_output, device_output_fp16, B*M*H_out*W_out);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    cudaFree(device_input_fp16);
    cudaFree(device_mask_fp16);
    cudaFree(device_output_fp16);

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
