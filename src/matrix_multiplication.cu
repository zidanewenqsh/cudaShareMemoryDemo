#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>
#define BLOCK_SIZE 16

__global__ void MatrixMulShared(float *A, float *B, float *C, int widthA, int heightA, int widthB, int heightB) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0;

    for (int m = 0; m < (widthA + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (Row < heightA && m * BLOCK_SIZE + tx < widthA) {
            sA[ty][tx] = A[Row * widthA + m * BLOCK_SIZE + tx];
        } else {
            sA[ty][tx] = 0.0;
        }

        if (Col < widthB && m * BLOCK_SIZE + ty < heightB) {
            sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * widthB + Col];
        } else {
            sB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (Row < heightA && Col < widthB) {
        C[Row * widthB + Col] = Cvalue;
    }
}

// 假设我们有两个矩阵，大小为 4x4
#define WIDTH_A 4
#define HEIGHT_A 4
#define WIDTH_B 4
#define HEIGHT_B 4

// // 前面定义的核函数
// __global__ void MatrixMulShared(float *A, float *B, float *C, int widthA, int heightA, int widthB, int heightB);

int main() {
    float *A, *B, *C; // 主机内存中的矩阵
    float *d_A, *d_B, *d_C; // 设备内存中的矩阵

    // 为矩阵分配主机内存
    A = (float *)malloc(WIDTH_A * HEIGHT_A * sizeof(float));
    B = (float *)malloc(WIDTH_B * HEIGHT_B * sizeof(float));
    C = (float *)malloc(WIDTH_B * HEIGHT_A * sizeof(float));

    // 初始化矩阵A和B
    for (int i = 0; i < WIDTH_A * HEIGHT_A; ++i) {
        A[i] = i;
    }
    for (int i = 0; i < WIDTH_B * HEIGHT_B; ++i) {
        B[i] = i;
    }

    // 为矩阵在设备上分配内存
    cudaMalloc((void **)&d_A, WIDTH_A * HEIGHT_A * sizeof(float));
    cudaMalloc((void **)&d_B, WIDTH_B * HEIGHT_B * sizeof(float));
    cudaMalloc((void **)&d_C, WIDTH_B * HEIGHT_A * sizeof(float));

    // 将矩阵从主机复制到设备
    cudaMemcpy(d_A, A, WIDTH_A * HEIGHT_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, WIDTH_B * HEIGHT_B * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块大小和网格大小
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((WIDTH_B + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT_A + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用核函数
    // auto start = std::chrono::high_resolution_clock::now();
    MatrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, WIDTH_A, HEIGHT_A, WIDTH_B, HEIGHT_B);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Total Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    
        // 记录结束时间
    cudaEventRecord(stop);

    // 等待事件完成
    cudaEventSynchronize(stop);

    // 计算运行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed Time: " << milliseconds << " ms" << std::endl;
    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, WIDTH_B * HEIGHT_A * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < HEIGHT_A; i++) {
        for (int j = 0; j < WIDTH_B; j++) {
            std::cout << C[i * WIDTH_B + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放资源
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
