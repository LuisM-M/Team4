// Source: AI
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <fstream>

double start, end;
const char *data_init = "data_init"; //
const char *mpi_send = "mpi_send";//
const char *mpi_recv = "mpi_recv";//
const char *comp = "comp"; //
const char *comp_small = "comp_small";//
const char *comp_large = "comp_large";//

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

void countSort(int arr[], int n, int exp) {
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
}

// Radix Sort Function
void radixSort(int arr[], int n) {
    CALI_MARK_BEGIN(comp_large);
    int maxValue = getMaxValue(arr, n);
    for (int exp = 1; maxValue / exp > 0; exp *= 10) {
        CALI_MARK_BEGIN(comp_small);
        countSort(arr, n, exp);
        CALI_MARK_END(comp_small);
    }
    CALI_MARK_END(comp_large);
}

int main(int argc, char** argv) {
    // Initialization and argument checking remains the same
    // MARK FUNCTION
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    CALI_MARK_BEGIN(data_init);
    MPI_Init(&argc, &argv);
    int num_vals, rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    num_vals = atoi(argv[1]); // Assuming num_vals is perfectly divisible by comm_size
    int local_num_vals = num_vals / comm_size;
    int *local_arr = new int[local_num_vals];
    int *arr = new int[num_vals];
    std::string input_type = argv[2];
    
    if (rank == 0) {
        // Initialize and populate arr[]...
        std::cout << "input type : "<< input_type << std::endl;
        if (argc < 4) {
            for(int i = 0; i < num_vals; i++) {
                arr[i] = (rand() % 10000) + 1;
            }
        }
        else {
            std::cout << "FILE READING" << std::endl;
            std::string fname = argv[3];
            std::ifstream file(fname);
            std::cout << "FILE NAME: " << fname << std::endl;
            if (!file.is_open()) {
                std::cerr << "Error opening file!" << std::endl;
            }

            for (int i = 0; i < num_vals; i++) {
                file >> arr[i];
            }
            file.close();
        }
    }

    MPI_Bcast(arr, num_vals, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(mpi_send);
    MPI_Scatter(arr, local_num_vals, MPI_INT, local_arr, local_num_vals, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_send);

    CALI_MARK_BEGIN(comp); // may need to move
    start = MPI_Wtime();
    radixSort(local_arr, local_num_vals);
    end = MPI_Wtime();
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(mpi_recv);
    MPI_Gather(local_arr, local_num_vals, MPI_INT, arr, local_num_vals, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_recv);

    if (rank == 0) {
        // Reapply the radix sort to the entire array
        for (int exp = 1; getMaxValue(arr, num_vals) / exp > 0; exp *= 10) {
            countSort(arr, num_vals, exp);
        }

        // // Display the sorted array
        // std::cout << "Sorted array: ";
        // for (int i = 0; i < num_vals; i++) {
        //     std::cout << arr[i] << " ";
        // }
        // std::cout << std::endl;
        // check sorted array
        bool sorted = true;
        for (int i = 1; i < num_vals; i++) {
            if (arr[i] < arr[i-1]) {
                std::cout << "NOT SORTED" << std::endl;
                sorted = false;
                break;
            }
        }
        if (sorted) {
            std::cout << "Sorted" << std::endl;
        }

        // Timing information
        std::cout << "Time taken for sorting: " << end - start << " seconds" << std::endl;
    }


    delete[] local_arr;
    delete[] arr;


    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Radix"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", num_vals); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", comm_size); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", 0); // The number of CUDA blocks 
    adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    MPI_Finalize();
    mgr.stop();
    mgr.flush();

    return 0;
}
