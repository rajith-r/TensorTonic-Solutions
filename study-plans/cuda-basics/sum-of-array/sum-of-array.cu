#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    // Write code here
    // __shared__ float* tile[blockDim.x];

    int i = blockDim.x*blockIdx.x+threadIdx.x;
    while(i<((N+1)/2)){
        int left = 2*i;
        int right = left+1;
        float curr_sum = input[left];
        if(right<N){
            curr_sum += input[right];
        }
        result[i] = curr_sum;
        i+=gridDim.x*blockDim.x;
    }
    // if(i<threadIdx.x){
    //     tile[threadIdx.x]=i
    // }
    // __sync_threads();

    // for(int i=threadIdx.x)
    // results[blockIdx.x*blockDim.x] = tile[0];

}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int currN = N;
    int blocks = (currN + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    
    float*  bufA;
    cudaMalloc(&bufA,sizeof(float)*N);
    float*  bufB;
    cudaMalloc(&bufB,sizeof(float)*N);
    sum_kernel<<<blocks, threads>>>(input, bufA, currN);
    

    float* src = bufA;
    float* dst = bufB;
    if (N <= 0) {
        return;
    }

    if (N == 1) {
        cudaMemcpy(
            result,
            input,
            sizeof(float),
            cudaMemcpyDeviceToDevice
        );
        return;
    }
    while(currN>1){
        currN=(currN+1)/2;
        blocks = (currN + threads - 1) / threads;
        if(currN == 1){
            sum_kernel<<<blocks, threads>>>(src, result, currN);
        }else{
            sum_kernel<<<blocks, threads>>>(src, dst, currN);
            src = dst;
            dst = (dst == bufA)?bufB:bufA;
        }
    }
    
    cudaDeviceSynchronize();
}
