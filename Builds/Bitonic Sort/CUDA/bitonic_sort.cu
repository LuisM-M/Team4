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

float effective_bandwidth_gb_s = 0.0;
  float bitonic_sort_step_time = 0.0;
  float cudaMemcpy_host_to_device_time = 0.0;
  float cudaMemcpy_device_to_host_time = 0.0;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
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
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float); 
  
  cudaEvent_t startEvent1, stopEvent1; //
  cudaEventCreate(&startEvent1); //
  cudaEventCreate(&stopEvent1); //
  

  cudaMalloc((void**) &dev_values, size);
  
  //cudaEventRecord(startEvent1); // 
  //MEM COPY FROM HOST TO DEVICE ********************************************************
  
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(startEvent1); 
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  //CALI_MARK_END(cudaMemcpy_host_to_device);
  
  cudaEventRecord(stopEvent1); //
  cudaEventSynchronize(stopEvent1); //
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, startEvent1, stopEvent1); //
  
  CALI_MARK_END(cudaMemcpy_host_to_device);
  //cudaDeviceSynchronize();


  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  /* Major step */
  
  // BITONC SORT ******************************************************************************
  
  cudaEvent_t startEvent2, stopEvent2; //
  cudaEventCreate(&startEvent2); //
  cudaEventCreate(&stopEvent2); //
  
  CALI_MARK_BEGIN(bitonic_sort_step_region);
  cudaEventRecord(startEvent2); // 
  
  
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      numiter++;
    }
  }
  CALI_MARK_END(bitonic_sort_step_region);
  cudaDeviceSynchronize();
  
  cudaEventRecord(stopEvent2); //
  cudaEventSynchronize(stopEvent2); //
  cudaEventElapsedTime(&bitonic_sort_step_time, startEvent2, stopEvent2); //
  
  //MEM COPY FROM DEVICE TO HOST ************************************************************************
  cudaEvent_t startEvent3, stopEvent3; //
  cudaEventCreate(&startEvent3); //
  cudaEventCreate(&stopEvent3); //
  
  cudaEventRecord(startEvent3); // 
  
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  
  //CALI_MARK_END(cudaMemcpy_device_to_host);
  
  cudaEventRecord(stopEvent3); //
  cudaEventSynchronize(stopEvent3); //
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, startEvent3, stopEvent3); //
  CALI_MARK_END(cudaMemcpy_device_to_host);
  //cudaDeviceSynchronize();
  cudaFree(dev_values);
  
  
}

int main(int argc, char *argv[])
{
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

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
  
  effective_bandwidth_gb_s = (NUM_VALS*4*4 *numiter / 1e9 ) / (bitonic_sort_step_time/ 1000);

  // Store results in these variables.
 // float effective_bandwidth_gb_s;
  //float bitonic_sort_step_time;
  //float cudaMemcpy_host_to_device_time;
 // float cudaMemcpy_device_to_host_time;
  printf("Elapsed time (cudaMemcpy host to device ):  %f\n", cudaMemcpy_host_to_device_time/1000);
  printf("Elapsed time (cudaMemcpy device to host ):  %f\n", cudaMemcpy_device_to_host_time/1000);
  printf("Elapsed time (bitonic_sort): %f\n", bitonic_sort_step_time/1000);
  printf("Calculated bandwidth):  %f\n", effective_bandwidth_gb_s );

  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("datatype_size", sizeof(float));
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}