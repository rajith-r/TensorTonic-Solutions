#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    // Write code here
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    int end = (N+1)/2;
    while(i<end){
        int left = 2*i;
        int right = left+1;
        float sum = input[left];

        if(right<N){
            sum+=input[right];
        }
        result[i] = sum;
        i+=stride;
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));

    float* bufA;
    cudaMalloc(&bufA,sizeof(float)*N);

    float* bufB;
    cudaMalloc(&bufB,sizeof(float)*N);

    sum_kernel<<<blocks, threads>>>(input, bufA, N);
    float* src;
    float* dst;
    int tempN = N;
    
    src = bufA;
    dst = bufB;
    if(N == 0){ 
        return;
    }else if(N == 1){
        cudaMemcpy(result, input, sizeof(float), cudaMemcpyDeviceToDevice);
    }else{
        while(tempN>1){
            tempN = (tempN+1)/2;
            blocks = (tempN + threads - 1) / threads;
            if(tempN == 1){
                sum_kernel<<<blocks, threads>>>(src, result, tempN);
            }else{
                sum_kernel<<<blocks, threads>>>(src, dst, tempN);
                src = dst;
                (dst ==  bufA)?bufB:bufA;   
            }
        }
    }
    cudaDeviceSynchronize();
}
