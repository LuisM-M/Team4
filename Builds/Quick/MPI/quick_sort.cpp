// C++ program to implement the Quick Sort Algorithm using MPI
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <unistd.h>

// Function to swap two numbers
void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

// Function that performs the Quick Sort
void quicksort(int* arr, int start, int end) {
    if (end <= 1)
        return;

    // Pick pivot and swap with first element
    int pivot = arr[start + end / 2];
    swap(arr, start, start + end / 2);

    // Partitioning Steps
    int index = start;
    for (int i = start + 1; i < start + end; i++) {
        if (arr[i] < pivot) {
            index++;
            swap(arr, i, index);
        }
    }

    // Swap the pivot into place
    swap(arr, start, index);

    // Recursive Call for sorting
    quicksort(arr, start, index - start);
    quicksort(arr, index + 1, start + end - index - 1);
}

// Function that merges the two arrays
int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* result = new int[n1 + n2];
    int i = 0, j = 0, k = 0;

    while (i < n1 || j < n2) {
        if (i == n1) {
            result[k++] = arr2[j++];
        } else if (j == n2) {
            result[k++] = arr1[i++];
        } else if (arr1[i] < arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    return result;
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
	const char* whole_computation = "whole_computation";
    MPI_Init(&argc, &argv);

    int number_of_elements;
    int number_of_process, rank_of_process;
    double time_taken;

	// WHOLE PROGRAM COMPUTATION PART STARTS HERE
	CALI_MARK_BEGIN(whole_computation);
	double start_time_whole = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);

    int* data = nullptr;

    if (rank_of_process == 0) {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <inputfile> <outputfile>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::ifstream file(argv[1]);
        if (!file.is_open()) {
            std::cerr << "Error in opening file\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        file >> number_of_elements;
        int chunk_size = (number_of_elements + number_of_process - 1) / number_of_process;

        data = new int[number_of_process * chunk_size]();
        for (int i = 0; i < number_of_elements; ++i) {
            file >> data[i];
        }

        file.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time_taken = -MPI_Wtime();
	commTime = MPI_Wtime();
    CALI_MARK_BEGIN("commTime");
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = (number_of_elements + number_of_process - 1) / number_of_process;
    int* chunk = new int[chunk_size];

    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    delete[] data;
    data = nullptr;

    int own_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 1))
                             ? chunk_size
                             : (number_of_elements - chunk_size * rank_of_process);

    quicksort(chunk, 0, own_chunk_size);

    for (int step = 1; step < number_of_process; step *= 2) {
        if (rank_of_process % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT, rank_of_process - step, 0, MPI_COMM_WORLD);
            break;
        }
        if (rank_of_process + step < number_of_process) {
            int received_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 2 * step))
                                          ? (chunk_size * step)
                                          : (number_of_elements - chunk_size * (rank_of_process + step));

            int* chunk_received = new int[received_chunk_size];
            MPI_Recv(chunk_received, received_chunk_size, MPI_INT, rank_of_process + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int* temp = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);
            delete[] chunk;
            delete[] chunk_received;
            chunk = temp;
            own_chunk_size += received_chunk_size;
        }
    }
	// WHOLE PROGRAM COMPUTATION PART ENDS HERE
	CALI_MARK_END(whole_computation);
	end_time_whole = MPI_Wtime();

    adiak::init(NULL);
	adiak::launchdate();    // launch date of the job
	adiak::libraries();     // Libraries used
	adiak::cmdline();       // Command line used to launch the job
	adiak::clustername();   // Name of the cluster
	adiak::value("Algorithm", "Quick sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
	adiak::value("ProgrammingModel", MPI); // e.g., "MPI", "CUDA", "MPIwithCUDA"
	adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
	adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
	adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
	adiak::value("InputType", sorted); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
	adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
	adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
	adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
	adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
	adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

	// Flush Caliper output before finalizing MPI
	mgr.stop();
	mgr.flush();

	MPI_Finalize();
}
