#include <cuda_runtime.h>
#include <iostream>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 额外的 +1 用于避免共享内存的 bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    // int width_grid = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // 转置块的坐标
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
int main() {
    const int nx = 4;
    const int ny = 4;
    const int mem_size = nx * ny * sizeof(float);

    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);

    // 初始化输入数据
    for (int i = 0; i < nx * ny; ++i) {
        h_idata[i] = static_cast<float>(i);
    }

    float *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, mem_size);
    cudaMalloc((void **)&d_odata, mem_size);

    // 复制数据到设备
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((nx + TILE_DIM - 1) / TILE_DIM, (ny + TILE_DIM - 1) / TILE_DIM);

    // 执行核函数
    transpose<<<grid, block>>>(d_odata, d_idata, nx, ny);

    // 复制结果回主机
    cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (h_odata[x * ny + y] != h_idata[y * nx + x]) {
                printf("Error at %d,%d\n", y, x);
                success = false;
                break;
            }
        }
        if (!success) break;
    }
    if (success) printf("Transpose successful!\n");
    
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            std::cout << h_odata[i * nx + j] << " ";
        }
        std::cout << std::endl;
    }
    // 清理资源
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
