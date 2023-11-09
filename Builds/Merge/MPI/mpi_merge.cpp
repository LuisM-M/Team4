#include <iostream>
#include <cstdlib>
#include <cstring>
#include <mpi.h>


// Sources: geeksforgeeks.org/merge-sort

// Function to merge two sorted subarrays
void Merge(int* array, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; i++) {
        L[i] = array[left + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = array[middle + 1 + i];
    }

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            array[k] = L[i];
            i++;
        } else {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        array[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        array[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

// Traditional MergeSort algorithm
void MergeSort(int* array, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;
        MergeSort(array, left, middle);
        MergeSort(array, middle + 1, right);
        Merge(array, left, middle, right);
    }
}

// Function to merge all sorted subarrays collected at the root
int* MergeAllSortedSubArrays(int* sortedSubArrays, int numberOfSubArrays, int subArraySize) {
    // Sequential merge for simplicity
    int* mergedArray = new int[numberOfSubArrays * subArraySize];
    std::memcpy(mergedArray, sortedSubArrays, subArraySize * sizeof(int));

    for (int i = 1; i < numberOfSubArrays; i++) {
        Merge(mergedArray, 0, i * subArraySize - 1, (i + 1) * subArraySize - 1);
    }

    return mergedArray;
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dataSize = ...; // Set the size of the data to sort
    int* data = new int[dataSize]; // Initialize and populate data

    int subArraySize = dataSize / size;
    int* subArray = new int[subArraySize];

    MPI_Scatter(data, subArraySize, MPI_INT, subArray, subArraySize, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform the local sort
    MergeSort(subArray, 0, subArraySize - 1);

    int* sortedSubArrays = nullptr;
    if (rank == 0) {
        sortedSubArrays = new int[dataSize];
    }

    MPI_Gather(subArray, subArraySize, MPI_INT, sortedSubArrays, subArraySize, MPI_INT, 0, MPI_COMM_WORLD);

    int* sortedData = nullptr;
    if (rank == 0) {
        sortedData = MergeAllSortedSubArrays(sortedSubArrays, size, subArraySize);
        delete[] sortedSubArrays;
    }

    delete[] subArray;

    if (rank == 0) {
        // Do something with the sorted data in the root process
        delete[] sortedData;
    }

    delete[] data;

    MPI_Finalize();
    return 0;


    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "float");
    adiak::value("SizeOfDatatype", sizeof(float));
    // adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "AI"); 
}
