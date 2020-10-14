[TOC]

# CUDA

## Hard Device & abbreviation

SM : 流多处理器，相当于一个多线程的 CPU 核

SP : 实际处理单个硬件线程

线程与进程：对于操作系统来说，一个任务就是一个进程（Process）。在一个进程内部，要同时干多件事，就需要同时运行多个“子任务”，我们把进程内的这些“子任务”称为线程（Thread）。

## kernel function

```c
__global__ void square(float* d_dout, float* d_in){
    int idx = threadIdx.x;
}
```

threadIdx is a C struct, with `x, y, z` member. `x` denotes thread number.

### memory management

`cudaMalloc` 在显存中申请内存，与 C 中 `malloc` 相同。**一定要注意内存溢出情况**

### 线程网格、线程块、线程

`blockDim` 每个线程块中所有的线程数；`blockIdx`线程块的索引；`gridDim` 线程块在每个维度上的数目。`warpSize` 线程束中线程数目。 

* 线程块

```c
kernel_function<<<num_blocks, num_threads>>>(param1, param2, ...)

// calculate thread index
const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
```

所调用的总线程数为 `num_blocks * numthreads` 。`num_threads` 主要受硬件的 SM 限制。最大旧设备为 512，新设备为 1024。

* 线程网格

在两个维度上都有线程分布，处理二维数组使用。调度

```c
dim3 thread_rect(32, 4); // 每个线程块中线程分布 (x, y, z) 未指定的默认为 1
dim3 blocks_rect(1, 4);  // 线程块分布

kernel_function<<< dim3(bx, by, bz), dim3(tx, ty, tz), share_memory >>>(...);
```

线程索引

```c
const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

rect_array[idy][idx] = 1.0; // 二维数组索引赋值

const unsigned int thread_idx = (gridDim.x * blockDim.x) * idy + idx; \\ 线程数
```

* 线程数

线程束是`GPU`执行的基本单元。每一组线程或每个线程束中的线程同时执行。一般是**一次访存**获取一组指令，广播到整个线程中同时执行。线程束大小由硬件本身决定，即`warpSize`。但是，**硬件的调度是基于半个线程束**。所以在运行分支（if, else, for, while）时，尽量保持半个线程束中指令保持一致。

## CUDA 内存处理

### 缓存

缓存是为了解决`时间局部性`问题，即计算机会在短时间内多次访问相同的数据。`存储带宽(memory bandwith)`: 在一定时间内 `DRAM` 读出或写入的数据。`延迟(latency)`：响应一个获取内存的请求所花费的时间。

* GPU 缓存结构

| 存储类型 | 寄存器 | 共享内存 | 纹理内存 | 常量内存 | 全局内存 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 带宽 /s | 8Tb | 1.5 Tb | 200Mb | 200Mb | 200Mb |

GPU 中有一个共享的二级缓存。共享缓存允许处理器之间不需要每次通过全局内存进行通信，加快数据传输。

### 寄存器

每线程中每个浮点型变量会占用一个寄存器。每个 `SM` 可供使用的寄存器大小是有限的。合理开启线程，以使得各个线程利用的寄存器数目最大。

寄存器使用可以减少程序对主内存的读写操作，节约时间。需要设计变量在寄存器还是在主内存中。

在 CUDA 代码中

```c
u32 d_tmp; // 声明在主内存中 

__device__ static u32 d_tmp; // 声明在寄存器中
```

> 值得注意的是，最好不要对单一的存储地址开启多线程操作，因为操作的顺序是未知的，并且操作堵塞会浪费大量的时间。最好是声明为数组。

### Example & Problem

* 直方图统计

`原子操作`： 某个线程对数据进行修改时，其它低优先级的操作无法打断它。问题在于，若数据量多，区间少，则并行操作由于等待，效率会降低。

`共享内存`

```c
__shared__ unsigned int d_bin_data_shared[256];

__global__ void histogram(
    const unsigned int const * d_hist_data,
    unsigned int * const d_bin_data
) {
    /* thread id */
    const unsigned int idx = (blockId.x + blockDim.x) + threadId.x;
    const unsigned int idy = (blockId.y + blockDim.y) + threadId.y;
    const unsigned int thread_id = (gridDim.x * blockDim.x) * idy + idx;

    /* clear shared memory */
    d_bin_data_shared[threadIdx.x] = 0;

    // fetch data at 32 bits
    const unsigned int value_32 = d_hist_data[thread_id];

    // wait for all threads to update shared memory
    __syncthreads();

    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0x000000FF)) ]), 1);
    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0x0000FF00)) >> 8]), 1);
    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0x00FF0000)) >> 16]), 1);
    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0xFF000000)) >> 24]), 1);
}

```

这里 4字节 读取，单字节处理，是为了将读取性能发挥到最大。

## CMAKE configuration

```cmake
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)

target_link_libraries(${Program} yololayer)
```
