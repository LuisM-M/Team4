/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

cudaEvent_t startEvent1, stopEvent1;

int THREADS;
int BLOCKS;
int NUM_VALS;
float numiter = 0;

// float effective_bandwidth_gb_s = 0.0;
//   float bitonic_sort_step_time = 0.0;
//   float cudaMemcpy_host_to_device_time = 0.0;
//   float cudaMemcpy_device_to_host_time = 0.0;

const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* data_init = "data_init";
const char* cudaMemcpy_1 = "cudaMemcpy_1";

// const char* bitonic_sort_step_region = "bitonic_sort_step";
// const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
// const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

int random_int()
{
  return (int)rand()/(int)RAND_MAX;
}

void array_print(int *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(int *arr, int length)
{
  CALI_MARK_BEGIN(data_init);
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_int();
  }
  CALI_MARK_END(data_init);
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

// correctness function
bool check_sorted(int *arr, int length) {
  for (int i = 0; i < length - 1; i++) {
    if (arr[i] > arr[i + 1]) {
      return false; // Array is not sorted
    }
  }
  return true; // Array is sorted
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values)
{
  int *dev_values;
  size_t size = NUM_VALS * sizeof(int); 

  cudaMalloc((void**) &dev_values, size);
  //MEM COPY FROM HOST TO DEVICE ********************************************************
  // comm
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);



  CALI_MARK_BEGIN(cudaMemcpy_1);
  
  // cuda memcpy
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  
  CALI_MARK_END(cudaMemcpy_1);
  
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  
  


  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  /* Major step */
  
  // BITONC SORT ******************************************************************************
  
  
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  
  
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      numiter++;
    }
  }
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);
  
  //MEM COPY FROM DEVICE TO HOST ************************************************************************
  
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  
  
  
}

int main(int argc, char *argv[])
{
  CALI_CXX_MARK_FUNCTION;
  // cali::ConfigManager mgr;
  // mgr.start();
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);
  

  
  
   

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  int *values = (int*) malloc( NUM_VALS * sizeof(int));
  array_fill(values, NUM_VALS);

  start = clock();
  bitonic_sort(values); // Sort the array

  
  stop = clock();
  // Check if the array is sorted **********************
  bool is_sorted = check_sorted(values, NUM_VALS);
  if (!is_sorted) {
    printf("The array is not sorted correctly.\n");
  } else {
    printf("The array is sorted correctly.\n");
  }

  print_elapsed(start, stop);
  
  // effective_bandwidth_gb_s = (NUM_VALS*4*4 *numiter / 1e9 ) / (bitonic_sort_step_time/ 1000);
  

  // Store results in these variables.
 // float effective_bandwidth_gb_s;
  //float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
 float cudaMemcpy_device_to_host_time;
  // printf("Elapsed time (cudaMemcpy host to device ):  %f\n", cudaMemcpy_host_to_device_time/1000);
  // printf("Elapsed time (cudaMemcpy device to host ):  %f\n", cudaMemcpy_device_to_host_time/1000);
  // printf("Elapsed time (bitonic_sort): %f\n", bitonic_sort_step_time/1000);
  // printf("Calculated bandwidth):  %f\n", effective_bandwidth_gb_s );

  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "BitonicSort");
  adiak::value("ProgrammingModel", "CUDA");
  adiak::value("Datatype", "int");
  adiak::value("SizeOfDatatype", sizeof(int));
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  // adiak::value("num_procs", "2"); // The number of processors (MPI ranks)
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("group_num", "4");
  adiak::value("implementation_source", "Handwritten"); 
  
  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

}