#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>
// #include <random>

const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comm_large = "comm_large";
const char *cudaMemc = "cudaMemcpy";

int THREADS;
int BLOCKS;
int NUM_VALS;

void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

// int generateRandomInt(int min, int max)
// {
//     static std::mt19937 rng(static_cast<unsigned int>(time(nullptr))); // Seed with time
//     std::uniform_int_distribution<int> dist(min, max);
//     return dist(rng);
// }

bool isSorted(int *array, int size)
{
    for (int i = 0; i < size - 1; ++i)
    {
        if (array[i] > array[i + 1])
        {
            return false;
        }
    }
    return true;
}

__global__ void mergeKernel(int *deviceArray, int *auxArray, int size, int width)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * width * 2;

    // Check if the thread is working within the bounds of the array
    if (start >= size)
        return;

    // Calculate the middle and end indices of the sections to merge
    int middle = min(start + width, size);
    int end = min(start + width * 2, size);

    // Merge the two halves
    int i = start, j = middle, k = start;
    while (i < middle && j < end)
    {
        if (deviceArray[i] < deviceArray[j])
        {
            auxArray[k++] = deviceArray[i++];
        }
        else
        {
            auxArray[k++] = deviceArray[j++];
        }
    }

    // Copy remaining elements from the left half
    while (i < middle)
    {
        auxArray[k++] = deviceArray[i++];
    }

    // Copy remaining elements from the right half
    while (j < end)
    {
        auxArray[k++] = deviceArray[j++];
    }
}

void cpuMerge(int *array, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temp arrays
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    if (!L || !R)
    {
        fprintf(stderr, "Memory allocation failed in cpuMerge.\n");
        free(L); // Free L in case it was allocated
        free(R); // Free R in case it was allocated
        return;  // Exit the function or handle the error as appropriate
    }

    // Copy data to temp arrays L[] and R[]
    for (int i = 0; i < n1; i++)
        L[i] = array[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = array[mid + 1 + j];

    // Merge the temp arrays back into array[left..right]
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            array[k] = L[i];
            i++;
        }
        else
        {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        array[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        array[k] = R[j];
        j++;
        k++;
    }

    // Free the temporary arrays
    free(L);
    free(R);
}

void mergeSort(int *hostArray, int size, int numThreads)
{
    int *deviceInputArray, *deviceSortedArray;
    size_t arraySizeInBytes = size * sizeof(int);
    cudaError_t err;

    err = cudaMalloc(&deviceInputArray, arraySizeInBytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        // Handle the error, e.g., by freeing resources and exiting
        return;
    }
    // cudaMalloc(&deviceInputArray, arraySizeInBytes);
    err = cudaMalloc(&deviceSortedArray, arraySizeInBytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        // Handle the error, e.g., by freeing resources and exiting
        return;
    }
    // cudaMalloc(&deviceSortedArray, arraySizeInBytes);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudaMemc);
    cudaMemcpy(deviceInputArray, hostArray, arraySizeInBytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudaMemc);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int subArraySize = 1024; // Adjust based on your requirements
    dim3 blockSize(numThreads);
    dim3 gridSize((subArraySize + blockSize.x - 1) / blockSize.x);
    BLOCKS = gridSize.x;

    printf("Kernel launch with subArraySize: %d, gridSize: %d, BLOCKS: %d\n", subArraySize, gridSize.x, BLOCKS);

    for (int width = 1; width < subArraySize; width *= 2)
    {
        for (int start = 0; start < size; start += subArraySize)
        {
            mergeKernel<<<gridSize, blockSize>>>(deviceInputArray + start, deviceSortedArray + start, subArraySize, width);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(deviceInputArray, deviceSortedArray, arraySizeInBytes, cudaMemcpyDeviceToDevice);
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudaMemc);
    cudaMemcpy(hostArray, deviceInputArray, arraySizeInBytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cudaMemc);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // CPU-based merge for larger sections
    for (int width = subArraySize; width < size; width *= 2)
    {
        for (int i = 0; i < size; i += 2 * width)
        {
            int mid = min(i + width - 1, size - 1);
            int right = min(i + 2 * width - 1, size - 1);
            cpuMerge(hostArray, i, mid, right);
        }
    }

    cudaFree(deviceInputArray);
    cudaFree(deviceSortedArray);
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    // BLOCKS = NUM_VALS / THREADS;

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    clock_t start, stop;

    CALI_MARK_BEGIN(data_init);
    int *array = (int *)malloc(NUM_VALS * sizeof(int));
    if (!array)
    {
        fprintf(stderr, "Memory allocation failed!");
        return 1;
    }

    // random input
    // for (int c = 0; c < NUM_VALS; c++)
    // {
    //     array[c] = generateRandomInt(0, NUM_VALS - 1);
    // }

    // sorted input
    // for (int i = 0; i < NUM_VALS; ++i)
    // {
    //     array[i] = i;
    // }

    // reversed input
    // for (int i = NUM_VALS - 1; i >= 0; i--)
    // {
    //     array[i] = i;
    // }

    // Perturb about 1% of the elements
    // srand(time(NULL));
    // for (int i = 0; i < NUM_VALS / 100; i++)
    // {
    //     int j = rand() % NUM_VALS;
    //     array[j] = rand(); // Use rand() to generate integer values
    // }
    CALI_MARK_END(data_init);

    start = clock();
    mergeSort(array, NUM_VALS, THREADS); // Corrected function name
    stop = clock();

    print_elapsed(start, stop);

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    CALI_MARK_BEGIN(correctness_check);
    if (isSorted(array, NUM_VALS))
    {
        printf("The array is correctly sorted.\n");
    }
    else
    {
        printf("The array is NOT correctly sorted.\n");
    }
    CALI_MARK_END(correctness_check);

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
    adiak::value("InputSize", NUM_VALS);
    adiak::value("InputType", "1%perturbed");
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "AI & Online");

    mgr.stop();
    mgr.flush();
    free(array);
    return 0;
}