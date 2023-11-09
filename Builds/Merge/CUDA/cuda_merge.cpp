#include <iostream>
#include <chrono>
#include <cstdlib>
#include <helper_cuda.h>
#include <sys/time.h>
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

//Sources: https://github.com/54kevinalbert/gpu-mergesort
/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its runnning time
 *  is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 *  running on the GPU may provide a significant boost in performance
 */


// data[], size, threads, blocks, 
void mergesort(long*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);


#define min(a, b) (a < b ? a : b)


int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

    CALI_MARK_BEGIN(main);

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    //
    // Parse argv
    //
    tm();
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'v':
                    break;
                default:
                    std::cout << "unknown argument: " << arg << '\n';
                    printHelp(argv[0]);
                    return -1;
            }

            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
        else {
            if (argv[i][0] == '?' && !argv[i][1])
                std::cout << "help:\n";
            else
                std::cout << "invalid argument: " << argv[i] << '\n';
            printHelp(argv[0]);
            return -1;
        }
    }



    //
    // Read numbers from stdin
    //
    long* data;
    long size = readList(&data);
    if (!size) return -1;


    // merge-sort the data
    auto start = std::chrono::steady_clock::now();
    mergesort(data, size, threadsPerBlock, blocksPerGrid);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	std::cout << "Merge Sort CUDA Time: " << end.count() << " ms" << std::endl;


    //
    // Print out the list
    //
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << '\n';
    } 

   
    CALI_MARK_END(main);

    cali::ConfigManager mgr;
	mgr.start();

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
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", etc.
    adiak::value("num_procs", 2); 
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "Online"); 

    mgr.stop();
    mgr.flush();
}

void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
     CALI_CXX_MARK_FUNCTION;

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
    cudaMalloc((void**) &D_data, size * sizeof(long));
    cudaMalloc((void**) &D_swp, size * sizeof(long));


    // Copy from our input list into the first array
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm)
 
    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commSmall);
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);
    CALI_MARK_END(commSmall);
    CALI_MARK_END(comm);



    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(compLarge);
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);


        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }
    CALI_MARK_END(compLarge);
    CALI_MARK_END(comp);

    //
    // Get the list back from the GPU
    //
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
    
    
    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// read data into a minimal linked list
typedef struct {
    int v;
    void* next;
} LinkNode;

// helper function for reading numbers from stdin
// it's 'optimized' not to check validity of the characters it reads in..
long readList(long** list) {
    long v, size = 0;
    LinkNode* node = 0;
    LinkNode* first = 0;
    while (std::cin >> v) {
        LinkNode* next = new LinkNode();
        next->v = v;
        if (node)
            node->next = next;
        else 
            first = next;
        node = next;
        size++;
    }


    if (size) {
        *list = new long[size]; 
        LinkNode* node = first;
        long i = 0;
        while (node) {
            (*list)[i++] = node->v;
            node = (LinkNode*) node->next;
        }

    }


    return size;
}


