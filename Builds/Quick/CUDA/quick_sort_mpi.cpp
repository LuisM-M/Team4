/**
 * -------------------- SOURCE -----------------------------------
 * Code: https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/#
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

// Function to swap two numbers
void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quicksort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

// Check for correctness
bool isSorted(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

// Driver Code
int main(int argc, char* argv[]) {
	CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    double startTotalTime, endTotalTime;
    double startCommTime, endCommTime;
    startTotalTime = MPI_Wtime();
    CALI_MARK_BEGIN("totalTime")
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]); // Size of the array
    int* sub_array = (int*)malloc((n / size) * sizeof(int));
    int* sorted = NULL;
    int* array = NULL;

    if (rank == 0) {
        array = (int*)malloc(n * sizeof(int));
        // Initialize the array with random data
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 100; // Random numbers between 0 and 99
        }
        sorted = (int*)malloc(n * sizeof(int));
    }
    // start communication computation
    CALI_MARK_BEGIN("commTime");
    startCommTime = MPI_Wtime();
    // Scatter the array to all processes
    MPI_Scatter(array, n / size, MPI_INT, sub_array, n / size, MPI_INT, 0, MPI_COMM_WORLD);

    quicksort(sub_array, 0, (n / size) - 1);

    // Gather the sorted sub_arrays back to the root process
    MPI_Gather(sub_array, n / size, MPI_INT, sorted, n / size, MPI_INT, 0, MPI_COMM_WORLD);
    // end communication computation
    CALI_MARK_END("commTime");
    double endCommTime = MPI_Wtime();
    if (rank == 0) {
        if (isSorted(sorted, n)) {
            printf("The array is correctly sorted.\n");
        } else {
            printf("The array is NOT correctly sorted.\n");
        }
        free(array);
        free(sorted);
    }

    free(sub_array);
	// WHOLE PROGRAM COMPUTATION PART ENDS HERE
	CALI_MARK_END(totalTime);
	endTotalTime = MPI_Wtime();

	double total_elapsed = endTotalTime - startTotalTime;
	double comm_elapsed = endCommTime - startCommTime;

    if (rank == 0) {
        printf("Whole computation time: %f seconds\n", total_elapsed);
        printf("Communication time: %f seconds\n", comm_elapsed);
    }

    adiak::init(NULL);
	adiak::launchdate();    // launch date of the job
	adiak::libraries();     // Libraries used
	adiak::cmdline();       // Command line used to launch the job
	adiak::clustername();   // Name of the cluster
	adiak::value("Algorithm", "Quick sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
	adiak::value("ProgrammingModel", MPI); // e.g., "MPI", "CUDA", "MPIwithCUDA"
	adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
	adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
	adiak::value("InputType", "random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
	adiak::value("num_procs", 4); // The number of processors (MPI ranks)
	//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
	//adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
	adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", AI) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

	// Flush Caliper output before finalizing MPI
	mgr.stop();
	mgr.flush();

	MPI_Finalize();
    return 0;
}
