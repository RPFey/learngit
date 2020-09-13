#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000;

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
    const unsigned int value_u32 = d_hist_data[thread_id];

    // wait for all threads to update shared memory
    __syncthreads();

    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0x000000FF)) ]), 1);
    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0x0000FF00)) >> 8]), 1);
    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0x00FF0000)) >> 16]), 1);
    atomicAdd(&(d_bin_data_shared[ ((value_u32 & 0xFF000000)) >> 24]), 1);
}

void generate_data(unsigned int* hist){
    for(int i = 0; i < ARRAY_SIZE; i++){
        hist[i] = random() % 256;
    }
}

int main() {
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    unsigned int hist_data[ARRAY_BYTES];
    generate_data(hist_data);

    return 0;
}

