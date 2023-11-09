#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda.h>

int array_with_values[8] = {10, 9, 39, 2, 74, 833, 903, 81};

// Function to swap two elements (in device code)
__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partitioning function (in device code)
__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// Simple QuickSort algorithm (in device code)
__global__ void quickSort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Recursively sort elements before partition and after partition
        quickSort<<<1, 1>>>(arr, low, pi - 1);
        quickSort<<<1, 1>>>(arr, pi + 1, high);
    }
}

// Host main function
int main() {
	CALI_CXX_MARK_FUNCTION;
    int n = sizeof(h_array) / sizeof(h_array[0]);

    int* d_array;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    CALI_MARK_BEGIN("total comp");
	start = clock();
    quickSort<<<1, 1>>>(d_array, 0, n - 1);
	stop = clock();
	CALI_MARK_END("total comp");
    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
	
	printf("Elapsed time: %.3fs\n:", ((double) (stop - start)) / CLOCKS_PER_SEC);
    // Output the sorted array
    for (int i = 0; i < n; i++)
        std::cout << h_array[i] << " ";
    std::cout << std::endl;

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

    return 0;
}
