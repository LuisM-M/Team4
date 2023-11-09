#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda.h>

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__device__ int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            // Swap arr[i] and arr[j]
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
    // Swap the pivot to the correct location
    int t = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = t;
    return i + 1;
}

__device__ void quicksort(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

__global__ void quicksortKernel(int *arr, int low, int high) {
    quicksort(arr, low, high);
}

bool isSorted(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    CALI_CXX_MARK_FUNCTION;
    if (argc != 3) {
        printf("Usage: %s <number of threads per block> <number of values>\n", argv[0]);
        exit(1);
    }
    clock_t start, stop;
    num_threads = atoi(argv[1]);
    num_vals = atoi(argv[2]);
    num_blocks = num_vals / num_threads;
    int *array;
    int *d_array; // Device array pointer

    start = clock();
    array = (int *)malloc(num_vals * sizeof(int));
    for (int i = 0; i < n; i++) {
        array[i] = rand() % 100;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_array, num_vals * sizeof(int));

    // Copy array from host to device
    cudaMemcpy(d_array, array, num_vals * sizeof(int), cudaMemcpyHostToDevice);

    cali::ConfigManager mgr;
    mgr.start();

    // Launch the quicksort kernel
    quicksortKernel<<<num_blocks, num_threads>>>(d_array, 0, num_vals - 1);
    cudaDeviceSynchronize();

    // Copy the sorted array back to the host
    cudaMemcpy(array, d_array, num_vals * sizeof(int), cudaMemcpyDeviceToHost);
    stop = clock();
    cudaFree(d_array);

    mgr.stop();
    mgr.flush();
    
    print_elapsed(start, stop);

    if (isSorted(array, num_vals)) {
        printf("The array is sorted.\n");
    } else {
        printf("The array is NOT sorted.\n");
    }

    adiak::init(NULL);
 	adiak::launchdate();    // launch date of the job
 	adiak::libraries();     // Libraries used
 	adiak::cmdline();       // Command line used to launch the job
 	adiak::clustername();   // Name of the cluster
 	adiak::value("Algorithm", "Quick sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
 	adiak::value("ProgrammingModel", CUDA); // e.g., "MPI", "CUDA", "MPIwithCUDA"
 	adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
 	adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
 	adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
 	adiak::value("InputType", sorted); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
 	adiak::value("num_procs", random); // The number of processors (MPI ranks)
 	adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
 	adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
 	adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
 	adiak::value("implementation_source", AI) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    free(array);
    return 0;
}