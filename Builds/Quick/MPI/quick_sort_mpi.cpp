#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

using namespace std;

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

int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr, i, j);
        }
    }
    swap(arr, i, high);
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
    // Scatter the array to all processes
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

    free(sub_array);

    adiak::init(NULL);
	adiak::launchdate();    // launch date of the job
	adiak::libraries();     // Libraries used
	adiak::cmdline();       // Command line used to launch the job
	adiak::clustername();   // Name of the cluster
	adiak::value("Algorithm", "Quick sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
	adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
	adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
	adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("InputSize", n); // The number of elements in input dataset (1000)
	adiak::value("InputType", "random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
	adiak::value("num_procs", size); // The number of processors (MPI ranks)
	//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
	//adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
	adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

	// Flush Caliper output before finalizing MPI
	mgr.stop();
	mgr.flush();

	MPI_Finalize();
    return 0;
}


// // Function to swap two numbers
// void swap(int* arr, int i, int j)
// {
//     int t = arr[i];
//     arr[i] = arr[j];
//     arr[j] = t;
// }
 
// // Function that performs the Quick Sort
// // for an array arr[] starting from the
// // index start and ending at index end
// void quicksort(int* arr, int start, int end)
// {
//     int pivot, index;
 
//     // Base Case
//     if (end <= 1)
//         return;
 
//     // Pick pivot and swap with first
//     // element Pivot is middle element
//     pivot = arr[start + end / 2];
//     swap(arr, start, start + end / 2);
 
//     // Partitioning Steps
//     index = start;
 
//     // Iterate over the range [start, end]
//     for (int i = start + 1; i < start + end; i++) {
 
//         // Swap if the element is less
//         // than the pivot element
//         if (arr[i] < pivot) {
//             index++;
//             swap(arr, i, index);
//         }
//     }
 
//     // Swap the pivot into place
//     swap(arr, start, index);
 
//     // Recursive Call for sorting
//     // of quick sort function
//     quicksort(arr, start, index - start);
//     quicksort(arr, index + 1, start + end - index - 1);
// }
 
// // Function that merges the two arrays
// int* merge(int* arr1, int n1, int* arr2, int n2)
// {
//     int* result = (int*)malloc((n1 + n2) * sizeof(int));
//     int i = 0;
//     int j = 0;
//     int k;
 
//     for (k = 0; k < n1 + n2; k++) {
//         if (i >= n1) {
//             result[k] = arr2[j];
//             j++;
//         }
//         else if (j >= n2) {
//             result[k] = arr1[i];
//             i++;
//         }
 
//         // Indices in bounds as i < n1
//         // && j < n2
//         else if (arr1[i] < arr2[j]) {
//             result[k] = arr1[i];
//             i++;
//         }
 
//         // v2[j] <= v1[i]
//         else {
//             result[k] = arr2[j];
//             j++;
//         }
//     }
//     return result;
// }
 
// // Driver Code
// int main(int argc, char* argv[])
// {
//     CALI_CXX_MARK_FUNCTION;
//     cali::ConfigManager mgr;
//     mgr.start();

//     int number_of_elements;
//     int* data = NULL;
//     int chunk_size, own_chunk_size;
//     int* chunk;
//     double time_taken;
//     MPI_Status status;
 
//     int number_of_process, rank_of_process;
//     int rc = MPI_Init(&argc, &argv);
 
//     if (rc != MPI_SUCCESS) {
//         printf("Error in creating MPI "
//                "program.\n "
//                "Terminating......\n");
//         MPI_Abort(MPI_COMM_WORLD, rc);
//     }
 
//     MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);
 
//     if (rank_of_process == 0) {
//         // Computing chunk size
//         chunk_size
//             = (number_of_elements % number_of_process == 0)
//                   ? (number_of_elements / number_of_process)
//                   : (number_of_elements / number_of_process
//                      - 1);
 
//         data = (int*)malloc(number_of_process * chunk_size
//                             * sizeof(int));
 
//         // Reading the rest elements in which
//         // operation is being performed
        
 
//         // Padding data with zero
//         for (int i = number_of_elements;
//              i < number_of_process * chunk_size; i++) {
//             data[i] = 0;
//         }
 
//         // Printing the array read from file
//         printf("Elements in the array is : \n");
//         for (int i = 0; i < number_of_elements; i++) {
//             printf("%d ", data[i]);
//         }
 
//         printf("\n");
 
//         fclose(file);
//         file = NULL;
//     }
 
//     // Blocks all process until reach this point
//     CALI_MARK_BEGIN(comm);
//     CALI_MARK_BEGIN(comm_large);
//     CALI_MARK_BEGIN(MPI_barrier);
//     MPI_Barrier(MPI_COMM_WORLD);
//     CALI_MARK_END(MPI_barrier);
//     CALI_MARK_END(comm_large);
//     CALI_MARK_END(comm);
 
//     // Starts Timer
//     time_taken -= MPI_Wtime();
 
//     // BroadCast the Size to all the
//     // process from root process
//     CALI_MARK_BEGIN(comm);
//     CALI_MARK_BEGIN(comm_large);
//     CALI_MARK_BEGIN(MPI_bcast);
//     MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     CALI_MARK_END(MPI_bcast);
//     CALI_MARK_END(comm_large);
//     CALI_MARK_END(comm);
 
//     // Computing chunk size
//     chunk_size
//         = (number_of_elements % number_of_process == 0)
//               ? (number_of_elements / number_of_process)
//               : number_of_elements
//                     / (number_of_process - 1);
 
//     // Calculating total size of chunk
//     // according to bits
//     chunk = (int*)malloc(chunk_size * sizeof(int));
 
//     // Scatter the chuck size data to all process
//     CALI_MARK_BEGIN(comm);
//     CALI_MARK_BEGIN(comm_large);
//     CALI_MARK_BEGIN(MPI_scatter);
//     MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
//     CALI_MARK_END(MPI_scatter);
//     CALI_MARK_END(comm_large);
//     CALI_MARK_END(comm);
//     free(data);
//     data = NULL;
 
//     // Compute size of own chunk and
//     // then sort them
//     // using quick sort
 
//     own_chunk_size = (number_of_elements
//                       >= chunk_size * (rank_of_process + 1))
//                          ? chunk_size
//                          : (number_of_elements
//                             - chunk_size * rank_of_process);
 
//     // Sorting array with quick sort for every
//     // chunk as called by process
//     quicksort(chunk, 0, own_chunk_size);
 
//     for (int step = 1; step < number_of_process;
//          step = 2 * step) {
//         if (rank_of_process % (2 * step) != 0) {
//             CALI_MARK_BEGIN(comm);
//             CALI_MARK_BEGIN(comm_small);
//             CALI_MARK_BEGIN(MPI_send);
//             MPI_Send(chunk, own_chunk_size, MPI_INT, rank_of_process - step, 0, MPI_COMM_WORLD);
//             CALI_MARK_END(MPI_send);
//             CALI_MARK_END(comm_small);
//             CALI_MARK_END(comm);
//             break;
//         }
 
//         if (rank_of_process + step < number_of_process) {
//             int received_chunk_size
//                 = (number_of_elements
//                    >= chunk_size
//                           * (rank_of_process + 2 * step))
//                       ? (chunk_size * step)
//                       : (number_of_elements
//                          - chunk_size
//                                * (rank_of_process + step));
//             int* chunk_received;
//             chunk_received = (int*)malloc(received_chunk_size * sizeof(int));
//             CALI_MARK_BEGIN(comm);
//             CALI_MARK_BEGIN(comm_small);
//             CALI_MARK_BEGIN(MPI_recv);
//             MPI_Recv(chunk_received, received_chunk_size,
//                      MPI_INT, rank_of_process + step, 0,
//                      MPI_COMM_WORLD, &status);
//             CALI_MARK_END(MPI_recv);
//             CALI_MARK_END(comm_small);
//             CALI_MARK_END(comm);

//             CALI_MARK_BEGIN(comm);
//             CALI_MARK_BEGIN(comm_large);
//             data = merge(chunk, own_chunk_size,
//                          chunk_received,
//                          received_chunk_size);
//             CALI_MARK_END(comm_large);
//             CALI_MARK_END(comm);
 
//             free(chunk);
//             free(chunk_received);
//             chunk = data;
//             own_chunk_size
//                 = own_chunk_size + received_chunk_size;
//         }
//     }
 
//     // Stop the timer
//     time_taken += MPI_Wtime();
 
//     // Opening the other file as taken form input
//     // and writing it to the file and giving it
//     // as the output
//     if (rank_of_process == 0) {
//         // Opening the file
//         file = fopen(argv[2], "w");
 
//         if (file == NULL) {
//             printf("Error in opening file... \n");
//             exit(-1);
//         }
 
//         // Printing total number of elements
//         // in the file
//         fprintf(
//             file,
//             "Total number of Elements in the array : %d\n",
//             own_chunk_size);
 
//         // Printing the value of array in the file
//         for (int i = 0; i < own_chunk_size; i++) {
//             fprintf(file, "%d ", chunk[i]);
//         }
 
//         // Closing the file
//         fclose(file);
 
//         printf("\n\n\n\nResult printed in output.txt file "
//                "and shown below: \n");
 
//         // For Printing in the terminal
//         printf("Total number of Elements given as input : "
//                "%d\n",
//                number_of_elements);
//         printf("Sorted array is: \n");
 
//         for (int i = 0; i < number_of_elements; i++) {
//             printf("%d ", chunk[i]);
//         }
 
//         printf(
//             "\n\nQuicksort %d ints on %d procs: %f secs\n",
//             number_of_elements, number_of_process,
//             time_taken);
//     }

//     adiak::init(NULL);
// 	adiak::launchdate();    // launch date of the job
// 	adiak::libraries();     // Libraries used
// 	adiak::cmdline();       // Command line used to launch the job
// 	adiak::clustername();   // Name of the cluster
// 	adiak::value("Algorithm", "Quick sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
// 	adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
// 	adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
// 	adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
// 	adiak::value("InputSize", number_of_elements); // The number of elements in input dataset (1000)
// 	adiak::value("InputType", "random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
// 	adiak::value("num_procs", 4); // The number of processors (MPI ranks)
// 	// adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
// 	// adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
// 	adiak::value("group_num", 4); // The number of your group (integer, e.g., 1, 10)
// 	adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

// 	// Flush Caliper output before finalizing MPI
// 	mgr.stop();
// 	mgr.flush();

//     MPI_Finalize();
//     return 0;
// }