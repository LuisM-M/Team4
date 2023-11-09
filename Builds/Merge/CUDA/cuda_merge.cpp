#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Define constants for array size and threads per block
const int ARRAY_SIZE = 1 << 20; // Example: 2^20 elements
const int THREADS_PER_BLOCK = 512;

// Utility function to check CUDA calls
inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
    }
    return err;
}

// Device function to merge two sorted halves of an array
__device__ void merge(int *data, int left, int middle, int right, int *temp) {
    int i = left;
    int j = middle;
    int k = left;

    while (i < middle && j < right) {
        if (data[i] < data[j]) {
            temp[k] = data[i];
            ++i;
        } else {
            temp[k] = data[j];
            ++j;
        }
        ++k;
    }

    while (i < middle) {
        temp[k] = data[i];
        ++i;
        ++k;
    }

    while (j < right) {
        temp[k] = data[j];
        ++j;
        ++k;
    }

    for (i = left; i < right; ++i) {
        data[i] = temp[i];
    }
}

// Kernel to perform parallel merging of blocks sorted by MergeSort
__global__ void kernel_mergeSort(int *data, int *temp, int arraySize, int width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int left = idx * width;
    int right = min(left + width, arraySize);
    int middle = min((left + right) / 2, right);

    if (middle > left) {
        merge(data, left, middle, right, temp);
    }
}

// Host function to call the CUDA kernel for sorting
void parallelMergeSort(int *data, int arraySize) {
    int *device_data;
    int *temp;

    // Allocate memory on the device
    checkCudaErr(cudaMalloc(&device_data, arraySize * sizeof(int)), "cudaMalloc device_data");
    checkCudaErr(cudaMalloc(&temp, arraySize * sizeof(int)), "cudaMalloc temp");

    // Copy data to the device
    checkCudaErr(cudaMemcpy(device_data, data, arraySize * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy to device_data");

    // Call kernel for each stage of the sort
    for (int width = 2; width <= arraySize; width *= 2) {
        int numBlocks = ceil(float(arraySize) / float(width * THREADS_PER_BLOCK));
        kernel_mergeSort<<<numBlocks, THREADS_PER_BLOCK>>>(device_data, temp, arraySize, width);
        checkCudaErr(cudaGetLastError(), "kernel_mergeSort launch failure");
        checkCudaErr(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel_mergeSort");
    }

    // Copy sorted data back to the host
    checkCudaErr(cudaMemcpy(data, device_data, arraySize * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy to host_data");

    // Free device memory
    cudaFree(device_data);
    cudaFree(temp);
}

int main() {
    CALI_CXX_MARK_FUNCTION;
    int *data = new int[ARRAY_SIZE];

    cali::ConfigManager mgr;
    mgr.start();

    // Initialize data to sort (for example, with random numbers)
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = rand() % ARRAY_SIZE;
    }

    // Sort the data
    parallelMergeSort(data, ARRAY_SIZE);

    // Check the data is sorted
    for (int i = 1; i < ARRAY_SIZE; ++i) {
        if (data[i] < data[i - 1]) {
            std::cerr << "Sort failed at index " << i << std::endl;
            break;
        }
    }

    // Cleanup
    delete[] data;

    return 0;

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("num_procs", 2); 
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "AI"); 

    mgr.stop();
    mgr.flush();
}
