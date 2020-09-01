# CUDA

Malloc memory in GPU

Memory Copy CPU -> GPU

launch threads(kernel)

Memory Copy GPU -> CPU

each thread runs a kernel function, which looks like a serial algorithm.

each thread knows their own index. (thread index)

```bash

```

## kernel function

```c
__global__ void square(float* d_dout, float* d_in){
    int idx = threadIdx.x;
}
```

threadIdx is a C struct, with `x, y, z` member. `x` denotes thread number.

### memory management

`cudaMalloc` 在显存中申请内存，与 C 中 `malloc` 相同。**一定要注意内存溢出情况**