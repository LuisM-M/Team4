// source: AI
#include <cuda.h>
#include <caliper/cali.h>
#include <iostream>
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <ctime>
#include <adiak.hpp>

const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* data_init = "data_init";
const char* cudaMemcpy_htd = "cudaMemcpy_htd";
const char* cudaMemcpy_dth = "cudaMemcpy_dth";

// Count digit occurrences kernel
__global__ void countDigitOccurrencesKernel(int* input, int* count, int numElements, int bit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numElements) {
        int digit = (input[index] >> bit) & 1;
        atomicAdd(&count[digit], 1);
    }
}

// Kernel for reordering elements based on computed positions
__global__ void reorderKernel(int* input, int* output, int* count, int* prefixSum, int numElements, int bit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numElements) {
        int digit = (input[index] >> bit) & 1;
        int pos = prefixSum[digit] + atomicAdd(&count[digit], 1);
        output[pos] = input[index];
    }
}

// Host function for radix sort
void radixSort(int* d_input, int* d_output, int numElements) {
    int* d_count;
    cudaMalloc(&d_count, 2 * sizeof(int));
    int* d_prefixSum;
    cudaMalloc(&d_prefixSum, 2 * sizeof(int));
    
    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);

    for (int bit = 0; bit < 32; ++bit) {
        cudaMemset(d_count, 0, 2 * sizeof(int));
        
        // Count digit occurrences
        countDigitOccurrencesKernel<<<gridSize, blockSize>>>(d_input, d_count, numElements, bit);
        cudaDeviceSynchronize();

        // Copy count to host and compute prefix sum
        int h_count[2];
        cudaMemcpy(h_count, d_count, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        thrust::exclusive_scan(h_count, h_count + 2, h_count);
        cudaMemcpy(d_prefixSum, h_count, 2 * sizeof(int), cudaMemcpyHostToDevice);

        // Reorder elements
        reorderKernel<<<gridSize, blockSize>>>(d_input, d_output, d_count, d_prefixSum, numElements, bit);
        cudaDeviceSynchronize();

        // Swap input and output for next iteration
        std::swap(d_input, d_output);
    }

    cudaFree(d_count);
    cudaFree(d_prefixSum);
}


// Include the radixSort function and other necessary kernels from previous steps

// Function to generate random array
void generateRandomArray(int* array, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 10000; // Random integers in the range [0, size)
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [numThreads] [arraySize]" << std::endl;
        return 1;
    }

    int numThreads = atoi(argv[1]);
    int arraySize = atoi(argv[2]);

    // Generate random array
    int* h_input = new int[arraySize];
    generateRandomArray(h_input, arraySize);

    // Allocate device memory
    int* d_input;
    cudaMalloc(&d_input, arraySize * sizeof(int));
    int* d_output;
    cudaMalloc(&d_output, arraySize * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, h_input, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Start Caliper instrumentation
    cali::Annotation("radix_sort").begin();

    // Perform radix sort
    radixSort(d_input, d_output, arraySize);

    // End Caliper instrumentation
    cali::Annotation("radix_sort").end();

    // Copy sorted data back to host
    cudaMemcpy(h_input, d_output, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Output results (for debugging, you may want to print only a few elements)
    for (int i = 0; i < 10; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Radix"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", arraySize); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 0); // The number of processors (MPI ranks)
    adiak::value("num_threads", numThreads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", arraySize/numThreads); // The number of CUDA blocks 
    adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    return 0;
}
