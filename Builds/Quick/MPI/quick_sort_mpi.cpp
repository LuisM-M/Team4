#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* MPI_recv = "MPI_Recv";
const char* MPI_bcast = "MPI_Bcast";
const char* MPI_scatter = "MPI_Scatter";
const char* MPI_send = "MPI_Send";
const char* MPI_barrier = "MPI_Barrier";
const char* MPI_gather = "MPI_Gather";
const char* correctness_check = "correctness_check";

// Function to swap two numbers
void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

bool isSorted(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

// Function that performs the Quick Sort
void quicksort(int* arr, int low, int high) {
    if (low < high) {
        int pivot = arr[low];
        int left = low + 1;
        int right = high;

        while (1) {
            while (left <= right && arr[left] <= pivot)
                left++;
            while (left <= right && arr[right] >= pivot)
                right--;
            if (left <= right) {
                swap(arr, left, right);
            }
            else {
                break;
            }
        }
        swap(arr, low, right);

        quicksort(arr, low, right - 1);
        quicksort(arr, right + 1, high);
    }
}

int main(int argc, char* argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]); // Size of the array

    int* sub_array = (int*)malloc((n / size) * sizeof(int));
    int* sorted = NULL;
    int* array = NULL;
    CALI_MARK_BEGIN(data_init);
    if (rank == 0) {
        array = (int*)malloc(n * sizeof(int));
        // Initialize the array with random data
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 100; // Random numbers between 0 and 99
        }
        sorted = (int*)malloc(n * sizeof(int));
    }
    CALI_MARK_END(data_init);

    // Start communication and computation

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_scatter);
    MPI_Scatter(array, n / size, MPI_INT, sub_array, n / size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPI_scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    quicksort(sub_array, 0, (n / size) - 1);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPI_barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_barrier);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_gather);
    MPI_Gather(sub_array, n / size, MPI_INT, sorted, n / size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(MPI_gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    if (rank == 0) {
        // Combine and sort the final result on rank 0
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        quicksort(sorted, 0, n - 1);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // Perform any further processing on the sorted array
    }

    free(sub_array);

    CALI_MARK_BEGIN(correctness_check);
    if (rank == 0) {
        if (isSorted(sorted, n)) {
            printf("The array is correctly sorted.\n");
        } else {
            printf("The array is NOT correctly sorted.\n");
        }
        free(array);
        free(sorted);
    }
    CALI_MARK_END(correctness_check);
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Quick sort"); // The name of the algorithm you are using
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes
    adiak::value("InputSize", n); // The number of elements in input dataset
    adiak::value("InputType", "random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1% perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("group_num", 4); // The number of your group (integer)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}