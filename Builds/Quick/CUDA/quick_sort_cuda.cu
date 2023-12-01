#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda.h>
#include <adiak.hpp>

int*	r_values;
int*	d_values;

const char *comm = "comm";
const char *comp = "comp";
const char *comm_large = "comm_large";
const char *comp_large = "comp_large";
const char *cuda_memcpy_h2d = "cuda_memcpy_h2d";
const char *cuda_memcpy_d2h = "cuda_memcpy_d2h";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";

 // initialize data set
void Init(int* values, int N, int size) {
	srand( time(NULL) );
    for (int x = 0; x < N; ++x) {
        values[x] = rand() % 100;
    }
    // for (int x = 0; x < N; ++x) {
    //     values[x] = x + 1;  // sorted
    // }
    // for (int x = 0; x < N; ++x) {
    //     values[x] = N - x - 1;  // reversed
    // }

    // int numElementsToSwitch = N / 100; // perturbed

    // for (int i = 0; i < numElementsToSwitch; ++i) {
    //     int index1 = rand() % size;
    //     int index2 = rand() % size;

    //     // Swap elements at index1 and index2
    //     int temp = values[index1];
    //     values[index1] = values[index2];
    //     values[index2] = temp;
    // }
}

// Kernel function
__global__ static void quicksort(int* values, int N) {
    #define MAX_LEVELS	300

	int pivot, L, R;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if(L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
	                        // swap start[idx] and start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // swap end[idx] and end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}    
 
 // program main
 int main(int argc, char **argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    int N = atoi(argv[1]);
    int MAX_THREADS = atoi(argv[2]);

	// printf("./quicksort starting with %d numbers...\n", N);
    // printf("./quicksort starting with %d MAXTHREAD...\n", MAX_THREADS);
 	size_t size = N * sizeof(int);
 	
 	// allocate host memory
 	r_values = (int*)malloc(size);

    cudaError_t err = cudaMalloc((void **)&d_values, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	// allocate threads per block
    const unsigned int cThreadsPerBlock = 128;
                
	//for (int i = 0; i < 5; ++i) {
    CALI_MARK_BEGIN(data_init);
    Init(r_values, N, size);
    CALI_MARK_END(data_init);

    // copy data to device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cuda_memcpy_h2d);
    cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(cuda_memcpy_h2d);
    cudaDeviceSynchronize();
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    printf("Beginning kernel execution...\n");

    // execute kernel
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values, N);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");  

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cuda_memcpy_d2h);
    cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cuda_memcpy_d2h);
    cudaDeviceSynchronize();
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // test
    printf("\nTesting results...\n");
    CALI_MARK_BEGIN(correctness_check);
    for (int x = 0; x < N - 1; x++) {
        if (r_values[x] > r_values[x + 1]) {
            printf("Sorting failed.\n");
            break;
        } else {
            if (x == N - 2)
                printf("SORTING SUCCESSFUL\n");
        }
    }
    CALI_MARK_END(correctness_check);
 	// free memory
 	free(r_values);
 	
 	cudaThreadExit();

    int num_blocks = N / MAX_THREADS;
    adiak::init(NULL);
 	adiak::launchdate();    // launch date of the job
 	adiak::libraries();     // Libraries used
 	adiak::cmdline();       // Command line used to launch the job
 	adiak::clustername();   // Name of the cluster
 	adiak::value("Algorithm", "Quick Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
 	adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
 	adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
 	adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
 	adiak::value("InputSize", N); // The number of elements in input dataset (1000)
 	adiak::value("InputType", "1% perturbed"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
 	// adiak::value("num_procs", ); // The number of processors (MPI ranks)
 	adiak::value("num_threads", MAX_THREADS); // The number of CUDA or OpenMP threads
 	adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
 	adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
 	adiak::value("implementation_source", "https://github.com/saigowri/CUDA/blob/master/quicksort.cu"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    mgr.stop();
    mgr.flush();

    return 0;
}