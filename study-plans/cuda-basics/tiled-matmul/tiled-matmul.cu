#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int  i = blockDim.y*blockIdx.y+threadIdx.y;
    int  j = blockDim.x*blockIdx.x+threadIdx.x;
    int  ty = threadIdx.y;
    int  tx = threadIdx.x;
    extern __shared__ float shared[];
    float* s_A = shared;
    float* s_B = shared + TILE_DIM*TILE_DIM;

    int phases = (K+TILE_DIM-1)/TILE_DIM;
    float sum =0.0f;
    for(int phase =0;phase<phases;phase++){
        int tilestart = phase*TILE_DIM;
        if(i < M && tilestart + tx < K){
            s_A[ty*TILE_DIM+tx] = A[i* K +(tx+tilestart)];
        }else{
            s_A[ty*TILE_DIM+tx] = 0.0f;
        }

        
        if(tilestart + ty < K && j < N){
            s_B[ty*TILE_DIM+tx] = B[(ty+tilestart)* N +j];  
        }else{
            s_B[ty*TILE_DIM+tx] = 0.0f;  
        }
        __syncthreads();
        
        for(int k=0;k<TILE_DIM;k++){
            sum+=s_A[ty*TILE_DIM+k]*s_B[k*TILE_DIM+tx];
        }
        __syncthreads();
    }

    if(i<M && j<N){
        C[i*N+j] = sum;
    }


    
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    size_t sharedBytes = TILE_DIM*TILE_DIM*2*sizeof(float);
    tiled_matmul_kernel<<<blocks,threads,sharedBytes>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}