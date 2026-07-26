#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Write code here
    int i = blockDim.y*blockIdx.y+threadIdx.y;
    int j = blockDim.x*blockIdx.x+threadIdx.x;

    if(i<M && j<N){
        float sum = 0.0;
        for(int k=0;k<K;k++){
           sum += A[i*K+k]*B[k*N+j];
        }
        C[i*N+j] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
