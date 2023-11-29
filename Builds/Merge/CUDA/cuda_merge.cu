// // Source Code from: https://github.com/Liadrinz/HelloCUDA/blob/master/cuda/MergeSort.cu by Cyang Liadrinz
// // I added the caliper & adiak implementations

// // 归并排序
// #include <stdio.h>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include <iostream>
// #include <stdlib.h>
// #include <time.h>
// #include <algorithm>
// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>

// const char *data_init = "data_init";
// const char *comm = "comm";
// const char *comm_small = "comm_small";
// const char *comm_large = "comm_large";
// const char *comp = "comp";
// const char *comp_small = "comp_small";
// const char *comp_large = "comp_large";
// const char *correctness_check = "correctness_check";

// bool isSorted(int *array, int size)
// {
//     for (int i = 0; i < size - 1; ++i)
//     {
//         if (array[i] > array[i + 1])
//         {
//             return false;
//         }
//     }
//     return true;
// }

// __global__ void MergeSort(int *nums, int *temp, int n)
// {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     for (int i = 2; i < 2 * n; i *= 2)
//     {
//         int len = i;
//         if (n - tid < len)
//             len = n - tid;
//         if (tid % i == 0)
//         {
//             int *seqA = &nums[tid], lenA = i / 2, j = 0;
//             int *seqB = &nums[tid + lenA], lenB = len - lenA, k = 0;
//             int p = tid;
//             while (j < lenA && k < lenB)
//             {
//                 if (seqA[j] < seqB[k])
//                 {
//                     temp[p++] = seqA[j++];
//                 }
//                 else
//                 {
//                     temp[p++] = seqB[k++];
//                 }
//             }
//             while (j < lenA)
//                 temp[p++] = seqA[j++];
//             while (k < lenB)
//                 temp[p++] = seqB[k++];
//             for (int j = tid; j < tid + len; j++)
//                 nums[j] = temp[j];
//         }
//         __syncthreads();
//     }
// }

// int main(int argc, char **argv)
// {
//     CALI_CXX_MARK_FUNCTION;
//     cali::ConfigManager mgr;
//     mgr.start();

//     // 初始化数列
//     int size = atoi(argv[1]); // 1024 * 1024;
//     int *nums = (int *)malloc(sizeof(int) * size);
//     srand(time(0));

//     CALI_MARK_BEGIN(data_init);
//     // random input
//     for (int i = 0; i < size; ++i)
//     {
//         nums[i] = rand() % size;
//     }

//     // sorted input
//     // for (int i = 0; i < size; ++i) {
//     //     nums[i] = i;
//     //     // printf("%d ", nums[i]);
//     // }

//     // reverse sorted input
//     // for (int i = 0; i < size; ++i)
//     // {
//     //     nums[i] = size - i;
//     // }
//     CALI_MARK_END(data_init);

//     // 拷贝到设备
//     int *dNums;
//     CALI_MARK_BEGIN(comm);
//     CALI_MARK_BEGIN(comm_large);
//     cudaMalloc((void **)&dNums, sizeof(int) * size);
//     cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToDevice);

//     // 临时存储
//     int *dTemp;
//     cudaMalloc((void **)&dTemp, sizeof(int) * size);
//     CALI_MARK_END(comm_large);
//     CALI_MARK_END(comm);
//     int num_threads = atoi(argv[2]);
//     printf("The size of the input is %d\n", size);
//     printf("The number of threads is %d\n", num_threads);

//     dim3 threadPerBlock(atoi(argv[2])); // it is 1024
//     dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x, 1, 1);

//     float blocknum = (size + num_threads - 1) / num_threads;

//     CALI_MARK_BEGIN(comp);
//     CALI_MARK_BEGIN(comp_large);
//     MergeSort<<<blockNum, threadPerBlock>>>(dNums, dTemp, size);
//     CALI_MARK_END(comp_large);
//     CALI_MARK_END(comp);

//     CALI_MARK_BEGIN(comm);
//     CALI_MARK_BEGIN(comm_large);
//     cudaMemcpy(nums, dNums, sizeof(int) * size, cudaMemcpyDeviceToHost);
//     CALI_MARK_END(comm_large);
//     CALI_MARK_END(comm);

//     // 打印结果
//     // for (int i = 0; i < size; ++i) {
//     //     printf("%d ", nums[i]);
//     // }
//     // printf("\n");
//     CALI_MARK_BEGIN(correctness_check);
//     bool sorting = std::is_sorted(nums, nums + size);
//     CALI_MARK_END(correctness_check);
//     if (sorting == true)
//     {
//         printf("\n");
//         printf("The array is correctly sorted.\n");
//         printf("\n");
//     }
//     else
//     {
//         printf("\n");
//         printf("The array is NOT correctly sorted.\n");
//         printf("\n");
//     }
//     printf("\n");
//     printf("\n");

//     free(nums);
//     cudaFree(dNums);
//     cudaFree(dTemp);

//     printf("Number of numbers: %d\n", size);
//     // printf("Sorting time: %fms\n", GetTickCount() - s);
//     mgr.stop();
//     mgr.flush();

//     adiak::init(NULL);
//     adiak::user();
//     adiak::launchdate();
//     adiak::libraries();
//     adiak::cmdline();
//     adiak::clustername();
//     adiak::value("Algorithm", "Merge Sort");
//     adiak::value("ProgrammingModel", "CUDA");
//     adiak::value("Datatype", "int");
//     adiak::value("SizeOfDatatype", "4");      // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
//     adiak::value("InputSize", size);          // The number of elements in input dataset (1000)
//     adiak::value("InputType", "Random");      // For sorting, this would be "Sorted", "ReverseSorted", "Random", etc.
//     adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
//     adiak::value("num_blocks", blocknum);     // The number of CUDA blocks
//     adiak::value("group_num", "4");
//     adiak::value("implementation_source", "Online");
// }

// Source Code from: https://github.com/chrishadi/cuda-sort/tree/main/CudaSort by Christian Hadirahardja
// I added the caliper & adiak implementations

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define OK 1
#define EXPECTATION_ERROR 1
#define MALLOC_ERROR 2
#define CUDA_ERROR 3

const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";
const char *correctness_check = "correctness_check";

typedef struct mergeSortResult
{
    cudaError_t cudaStatus;
    const char *msg;
} mergeSortResult_t;

cudaError_t mergeSortWithCuda(int *arr, unsigned int count);
bool assertArrEq(int *expected, int *actual, size_t size);
int testMergeSortWithCuda(int *actual, int *expected, const unsigned int count);

__global__ void mergeSortKernel(int *arr, int *aux, unsigned int blockSize, const unsigned int last)
{
    int x = threadIdx.x;
    int start = blockSize * x;
    int end = start + blockSize - 1;
    int mid = start + (blockSize / 2) - 1;
    int l = start, r = mid + 1, i = start;

    if (end > last)
    {
        end = last;
    }
    if (start == end || end <= mid)
    {
        return;
    }

    while (l <= mid && r <= end)
    {
        if (arr[l] <= arr[r])
        {
            aux[i++] = arr[l++];
        }
        else
        {
            aux[i++] = arr[r++];
        }
    }

    while (l <= mid)
    {
        aux[i++] = arr[l++];
    }
    while (r <= end)
    {
        aux[i++] = arr[r++];
    }

    for (i = start; i <= end; i++)
    {
        arr[i] = aux[i];
    }
}

inline mergeSortResult_t mergeSortError(cudaError_t cudaStatus, const char *msg)
{
    mergeSortResult_t error;
    error.cudaStatus = cudaStatus;
    error.msg = msg;
    return error;
}

inline mergeSortResult_t mergeSortSuccess()
{
    mergeSortResult_t success;
    success.cudaStatus = cudaSuccess;
    return success;
}

inline mergeSortResult_t doMergeSortWithCuda(int *arr, unsigned int count, int *dev_arr, int *dev_aux)
{
    const unsigned int last = count - 1;
    const unsigned size = count * sizeof(int);
    unsigned int threadCount;
    cudaError_t cudaStatus;
    char msg[1024];

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (unsigned int blockSize = 2; blockSize < 2 * count; blockSize *= 2)
    {
        threadCount = count / blockSize;
        if (count % blockSize > 0)
        {
            threadCount++;
        }

        // Launch a kernel on the GPU with one thread for each block.
        mergeSortKernel<<<1, threadCount>>>(dev_arr, dev_aux, blockSize, last);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            sprintf(msg, "mergeSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return mergeSortError(cudaStatus, msg);
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            sprintf(msg, "cudaDeviceSynchronize returned error code %d after launching mergeSortKernel!\n", cudaStatus);
            return mergeSortError(cudaStatus, msg);
        }
    }

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    return mergeSortSuccess();
}

cudaError_t mergeSortWithCuda(int *arr, unsigned int count)
{
    const unsigned int size = count * sizeof(int);
    int *dev_arr = 0;
    int *dev_aux = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for two vectors (main and aux array).
    cudaStatus = cudaMalloc((void **)&dev_arr, size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void **)&dev_aux, size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_arr);
        return cudaStatus;
    }

    mergeSortResult_t result = doMergeSortWithCuda(arr, count, dev_arr, dev_aux);

    if (result.cudaStatus != cudaSuccess)
    {
        fprintf(stderr, result.msg);
    }

    cudaFree(dev_arr);
    cudaFree(dev_aux);

    return cudaStatus;
}

int main()
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    const unsigned int count = 64;
    const unsigned int size = count * sizeof(int);
    int status = MALLOC_ERROR;
    int *actual = (int *)malloc(size);
    int *expected = (int *)malloc(size);

    if (actual != NULL && expected != NULL)
    {
        status = testMergeSortWithCuda(actual, expected, count);
    }
    else
    {
        fprintf(stderr, "malloc failed!");
    }

    free(actual);
    free(expected);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    int cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return CUDA_ERROR;
    }

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", "4");   // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size);       // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");   // For sorting, this would be "Sorted", "ReverseSorted", "Random", etc.
    adiak::value("num_threads", count);    // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", blockSize); // The number of CUDA blocks
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "Online");

    mgr.stop();
    mgr.flush();
    return status;
}

int cmpInt(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}

int testMergeSortWithCuda(int *actual, int *expected, const unsigned int count)
{
    CALI_MARK_BEGIN(data_init);
    for (unsigned int i = 0; i < count; i++)
    {
        expected[i] = actual[i] = rand();
    }
    CALI_MARK_END(data_init);

    qsort(expected, count, sizeof(int), cmpInt);

    cudaError_t cudaStatus = mergeSortWithCuda(actual, count);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "mergeSortWithCuda failed!");
        return CUDA_ERROR;
    }

    CALI_MARK_BEGIN("correctness_check");
    if (!assertArrEq(expected, actual, count * sizeof(int)))
    {
        puts("cuda sorted array is not equal to the qsorted array!");
        return EXPECTATION_ERROR;
    }
    puts("Array is sorted!");
    puts("test ok.");
    CALI_MARK_END("correctness_check");
    return OK;
}

bool assertArrEq(int *expected, int *actual, size_t size)
{
    return memcmp(expected, actual, size) == 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>
// #include <cuda_runtime.h>
// #include <cuda.h>

// /* Define Caliper region names */
// const char *mainFunction = "main";
// const char *data_init = "data_init";
// const char *correctness_check = "correctness_check ";
// const char *comm = "comm";
// const char *comm_large = "comm_large";
// const char *comm_small = "comm_small";
// const char *comp = "comp";
// const char *comp_large = "comp_large";
// const char *comp_small = "comp_small";

// __global__ void generateData(int *dataArray, int size, int inputType)
// {
//     int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     for (int i = 0; i < size; ++i)
//     {
//         dataArray[i] = rand() % size;
//     }
// }

// __device__ void merge(int *arr, int *temp, int left, int mid, int right)
// {
//     int i = left;
//     int j = mid + 1;
//     int k = left;

//     while (i <= mid && j <= right)
//     {
//         if (arr[i] <= arr[j])
//         {
//             temp[k++] = arr[i++];
//         }
//         else
//         {
//             temp[k++] = arr[j++];
//         }
//     }

//     while (i <= mid)
//     {
//         temp[k++] = arr[i++];
//     }

//     while (j <= right)
//     {
//         temp[k++] = arr[j++];
//     }

//     for (int i = left; i <= right; i++)
//     {
//         arr[i] = temp[i];
//     }
// }

// __global__ void mergeSort(int *arr, int *temp, int size)
// {
//     for (int currSize = 1; currSize <= size - 1; currSize = 2 * currSize)
//     {
//         for (int leftStart = 0; leftStart < size - 1; leftStart += 2 * currSize)
//         {
//             int mid = min(leftStart + currSize - 1, size - 1);
//             int rightEnd = min(leftStart + 2 * currSize - 1, size - 1);
//             merge(arr, temp, leftStart, mid, rightEnd);
//         }
//     }
// }

// void printArray(int *arr, int size)
// {
//     printf("Array: ");
//     for (int i = 0; i < size; i++)
//     {
//         printf("%d ", arr[i]);
//     }
//     printf("\n");
// }

// bool isSorted(int *arr, int size)
// {
//     for (int i = 0; i < size - 1; i++)
//     {
//         if (arr[i] > arr[i + 1])
//         {
//             return false;
//         }
//     }
//     return true;
// }

// int main(int argc, char **argv)
// {

//     if (argc != 4)
//     {
//         printf("Usage: %s <sorting_type> <num_processors> <num_elements>\n", argv[0]);
//         return 1;
//     }

//     int sortingType = atoi(argv[1]);
//     int numProcessors = atoi(argv[2]);
//     int numElements = atoi(argv[3]);

//     const char *sorting_type_name;
//     switch (sortingType)
//     {
//     case 0:
//         sorting_type_name = "Random";
//         break;
//     case 1:
//         sorting_type_name = "Sorted";
//         break;
//     case 2:
//         sorting_type_name = "ReverseSorted";
//         break;
//     default:
//         sorting_type_name = "Unknown";
//         break;
//     }

//     int *h_arr = new int[numElements];
//     int *d_arr;
//     int *temp;

//     CALI_MARK_BEGIN(mainFunction);
//     cali::ConfigManager mgr;
//     mgr.start();

//     CALI_MARK_BEGIN(data_init);

//     // Call generateData kernel to initialize array based on sorting type
//     int *d_generateResult;
//     cudaMalloc((void **)&d_generateResult, sizeof(int) * numElements);
//     generateData<<<(numElements + 255) / 256, 256>>>(d_generateResult, numElements, sortingType);
//     cudaMemcpy(h_arr, d_generateResult, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
//     cudaFree(d_generateResult);

//     CALI_MARK_END(data_init);

//     CALI_MARK_BEGIN(comm);
//     CALI_MARK_BEGIN(comm_large);
//     cudaMalloc((void **)&d_arr, sizeof(int) * numElements);
//     CALI_MARK_BEGIN(comm_small);
//     cudaMemcpy(d_arr, h_arr, sizeof(int) * numElements, cudaMemcpyHostToDevice);
//     CALI_MARK_END(comm_small);
//     cudaMalloc((void **)&temp, sizeof(int) * numElements);
//     CALI_MARK_END(comm_large);
//     CALI_MARK_END(comm);

//     // Print initial array if size is less than or equal to 32
//     // if (numElements <= 1024)
//     // {
//     //     printArray(h_arr, numElements);
//     // }

//     CALI_MARK_BEGIN(comp);
//     CALI_MARK_BEGIN(comp_large);
//     CALI_MARK_BEGIN(comp_small);
//     // Call mergeSort kernel
//     mergeSort<<<1, 1>>>(d_arr, temp, numElements);
//     cudaDeviceSynchronize();
//     CALI_MARK_END(comp_small);

//     CALI_MARK_END(comp_large);
//     CALI_MARK_END(comp);

//     // Print sorted array if size is less than or equal to 32
//     // if (numElements <= 1024)
//     // {
//     cudaMemcpy(h_arr, d_arr, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
//     //     printArray(h_arr, numElements);
//     // }

//     CALI_MARK_BEGIN(correctness_check);

//     // Check if the array is sorted
//     bool sorted = isSorted(h_arr, numElements);
//     if (sorted)
//     {
//         printf("Array is sorted.\n");
//     }
//     else
//     {
//         printf("Array is not sorted.\n");
//     }

//     CALI_MARK_END(correctness_check);

//     delete[] h_arr;
//     cudaFree(d_arr);
//     cudaFree(temp);

//     // Flush Caliper output
//     mgr.stop();
//     mgr.flush();

//     CALI_MARK_END(mainFunction);

//     adiak::init(NULL);
//     adiak::launchdate();
//     adiak::libraries();
//     adiak::cmdline();
//     adiak::clustername();
//     adiak::value("Algorithm", "Merge Sort");
//     adiak::value("ProgrammingModel", "CUDA");
//     adiak::value("Datatype", "int");
//     adiak::value("SizeOfDatatype", sizeof(int));
//     adiak::value("InputSize", numElements);
//     adiak::value("InputType", sorting_type_name);
//     adiak::value("num_processors", numProcessors);
//     adiak::value("group_num", 16);
//     adiak::value("implementation_source", "AI & Handwritten & Online");

//     return 0;
// }