// Source: AI

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>


#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

double start, end;
const char* radix_sort = "radix_sort";
const char* synchron = "synchron";
const char* gather = "gather";
// Get the maximum value in the array
int getMaxValue(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Radix Sort
void radixSort(int arr[], int n, int rank, int comm_size) {
    start = MPI_Wtime();
    // Radix begin 
    CALI_MARK_BEGIN(radix_sort);

    int maxValue = getMaxValue(arr, n);
    int exp = 1;

    while (maxValue / exp > 0) {
        int output[n];
        int count[10] = {0};

        for (int i = 0; i < n; i++) {
            count[(arr[i] / exp) % 10]++;
        }

        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        for (int i = n - 1; i >= 0; i--) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }

        for (int i = 0; i < n; i++) {
            arr[i] = output[i];
        }

        exp *= 10;

        // Synchronize all processes at each digit pass

        // Synchronize begin
        CALI_MARK_BEGIN(synchron);

        MPI_Barrier(MPI_COMM_WORLD);
        // Synchronize end
        CALI_MARK_END(synchron);

        // Each process sorts its part of the array
        int chunkSize = n / comm_size;
        int* localArr = new int[chunkSize];

        

        MPI_Scatter(arr, chunkSize, MPI_INT, localArr, chunkSize, MPI_INT, 0, MPI_COMM_WORLD);

        radixSort(localArr, chunkSize, rank, comm_size);

        // gather begin
        CALI_MARK_BEGIN(gather);

        MPI_Gather(localArr, chunkSize, MPI_INT, arr, chunkSize, MPI_INT, 0, MPI_COMM_WORLD);

        // gather begin
        CALI_MARK_END(gather);

        end = MPI_Wtime();

        delete[] localArr;
    }

    // Radix end
    CALI_MARK_END(radix_sort);
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    MPI_Init(&argc, &argv);
    int num_vals = atoi(argv[1]);
    int threads  = atoi(argv[2]);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Initialize the array (you should replace this with your data)

    unsigned int* arr = new unsigned int[num_vals];

    for(int i = 0; i < num_vals; i++) {
        arr[i] = (rand() % 10000) + 1;
    }

    // Ensure all processes have the same data
    MPI_Bcast(arr, num_vals, MPI_INT, 0, MPI_COMM_WORLD);

    radixSort(arr, num_vals, rank, comm_size);

    if (rank == 0) {
        std::cout << "Sorted array: ";
        for (int i = 0; i < n; i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Time taken for sorting: " << end - start << " seconds" << std::endl;
        std::cout << "Sorted array: ";
        for (int i = 0; i < n; i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }


    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries("MPI, Caliper");     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Radix"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", num_vals); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", comm_size); // The number of processors (MPI ranks)
    adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_vals/threads); // The number of CUDA blocks 
    adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    MPI_Finalize();

    delete[] arr;
    return 0;
}
