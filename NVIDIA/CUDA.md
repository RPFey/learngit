[TOC]

缩写说明：

SM : 流多处理器

# CUDA

Malloc memory in GPU

Memory Copy CPU -> GPU

launch threads(kernel)

Memory Copy GPU -> CPU

each thread runs a kernel function, which looks like a serial algorithm.

each thread knows their own index. (thread index)

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

`blockDim` 每个线程块所有的线程数；`blockIdx`线程块的索引；`gridDim` 线程块在每个维度上的数目。`warpSize` 线程束中线程数目。 

* 线程块

```c
kernel_function<<<num_blocks, num_threads>>>(param1, param2, ...)

// calculate thread index
const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
```

所调用的总线程数为 `num_blocks * numthreads` 。`num_threads` 主要受硬件的 SM 限制（一般为128）。

* 线程网格

在两个维度上都有线程分布，处理二维数组使用。调度

```c
dim3 thread_rect(32, 4); // 每个线程块中线程分布 4 (Y 方向) * 32 (X 方向) 
dim3 blocks_rect(1, 4);  // 线程块分布

kernel_function<<< block_rect, thread_rect >>>(...);
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
