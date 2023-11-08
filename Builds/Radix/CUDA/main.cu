// Author: Jack Liu
// Username: jackfly
// Source: https://github.com/jackfly/radix-sort-cuda

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string> 
#include <sstream>
#include <math.h>
#include <time.h>

#include "radix_sort.h"

// caliper includes
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

bool debug = true;

const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
const char* radix_region = "radix_region";

const char* algorithm = "Radix";
const char* programmingModel = "CUDA";
const char* dataType = "int";

void radixsort_gpu(unsigned int* h_in, unsigned int num, unsigned int threads, bool printArray = false)
{
    unsigned int* out_gpu = new unsigned int[num];
    
    unsigned int* d_in;
    unsigned int* d_out;

    cudaMalloc(&d_in, sizeof(unsigned int) * num);
    cudaMalloc(&d_out, sizeof(unsigned int) * num);

    // HTD Region
    // BEGIN
    CALI_MARK_BEGIN(cudaMemcpy_host_to_device);

    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num, cudaMemcpyHostToDevice);

    // END HTD REGION
    CALI_MARK_END(cudaMemcpy_host_to_device);

    // RADIX SORT REGION
    // BEGIN
    CALI_MARK_BEGIN(radix_region);

    radix_sort(d_out, d_in, num, threads);

    // END
    CALI_MARK_END(radix_region);

    // DTH Region
    // BEGIN
    CALI_MARK_BEGIN(cudaMemcpy_device_to_host);

    cudaMemcpy(out_gpu, d_out, sizeof(unsigned int) * num, cudaMemcpyDeviceToHost);
    
    // END
    CALI_MARK_END(cudaMemcpy_device_to_host);

    // check list sorting
    if(debug) {
      for(int i = 0; i < num; i++) {
        printf("%i ", out_gpu[i]);
      }
      printf("\n")
    }

    cudaFree(d_out);
    cudaFree(d_in);

    delete[] out_gpu;
}

int main(int argc, char** argv)
{
  CALI_CXX_MARK_FUNCTION;
  struct timespec start, stop;
  

  if(argc < 3) {
    printf("Error, not enough args");
    return 0;
  }
  
  int threads = atoi(argv[1]);
  int num_vals = atoi(argv[2]);
  int blocks = num_vals / threads;


  // create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  printf("num values: %d\n", num_vals);
  printf("num threads: %d\n", threads)

  // random values in array
  unsigned int* numbers = new unsigned int[num_vals];
  for(int i = 0; i < num_vals; i++) {
    numbers[i] = (rand() % 10000) + 1;
  }

  if(debug) {
    for(int i = 0; i < num_vals; i++) {
      printf("%i\n", numbers[i]);
    }
  }
  
  // sorting
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
  radixsort_gpu(numbers, num_vals, threads);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
  double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
  printf("@time of CUDA run:\t\t\t[%.3f] microseconds\n", dt);

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", num_vals); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", 0); // The number of processors (MPI ranks)
  adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", blocks); // The number of CUDA blocks 
  adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  delete[] numbers;


}