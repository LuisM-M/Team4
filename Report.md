# CSCE 435 Group project

## 1. Group members:
1. Emma Ong
2. Naimur Rahman
3. Anna Huang
4. Luis Martinez Morales

We will be communicating through Discord and text messaging.

---

## 2. _due 10/25_ Project topic

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

Merge Sort (MPI) *****************************************
1. Generate and populate an array with random numbers.
2. Initialize MPI for parallel processing.
3. Divide the array into equal-sized chunks based on the number of available processes.
4. Distribute these subarrays to different processes.
5. On each process, perform a local merge sort on its subarray.
6. Gather the sorted subarrays into one central location, typically on the root process.
7. On the root process:
   a. Perform a final merge sort on the gathered subarrays to create a fully sorted array.
   b. Display the sorted array.
8. Clean up resources, including memory used for subarrays and temporary arrays.
9. Finalize MPI to end parallel processing.

We’ll be leveraging sources for implementing Merge Sort with MPI like: https://www.geeksforgeeks.org/merge-sort/ , https://curc.readthedocs.io/en/latest/programming/MPI-C.html 

Merge Sort (MPI) *****************************************

Merge Sort (CUDA) *****************************************
1. Allocate GPU Memory: Reserve memory on the graphics processing unit (GPU) for the input list.
2. Copy to GPU: Transfer the original list from the CPU to the allocated GPU memory.
3. Determine Configuration: Decide how many threads to use in each GPU block and how many blocks based on the input list size and GPU capabilities.
4. Launch Merge Sort Kernel: Start the GPU process by running a CUDA kernel designed for parallel merge sorting.
5. Copy to CPU: Retrieve the sorted list from the GPU memory and place it back into the CPU memory.
6. Free GPU Memory: Release the GPU memory that was allocated for the input list.
7. Main Function:
    a. Read or generate the input list.
    b. Call the "ParallelMergeSort" function.
    c. Print the sorted list to the console.

We’ll be leveraging sources for implementing Merge Sort with CUDA like: https://www.geeksforgeeks.org/merge-sort/ , https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/ 

Radix Sort (CUDA): *********************************************************************************


function RadixSort(data):
    // First determine the number of elements
    numElements = length(data)

    // Allocate GPU memory for data and temp storage
    d_data = allocateGPUArray(numElements)

    d_temp = allocateGPUArray(numElements)

    // We must then copy the data from the CPU to GPU
    copyDataToGPU(data, d_data)

    // Determine the maximum number of bits needed for the data
    numBits = calculateMaxNumBits(data)

    // Perform radix sort using CUDA
    RadixSort(d_data, d_temp, numElements, numBits)

    // Copy the sorted data back from GPU to CPU
    sortedData = copyDataToCPU(d_data, numElements)

    // Free GPU memory
    freeGPUArray(d_data)
    freeGPUArray(d_temp)

    return sortedData
end function

For RadixSortCUDA function we plan to implement Radix Sort much like this reference from Geeks2Geeks https://www.geeksforgeeks.org/radix-sort/#:

function RadixSort(data, numElements):
    // Find the maximum value in the data to determine the number of digits
    maxVal = findMaxValue(data)
    numDigits = countDigits(maxVal)

    // Initialize a count array to hold the count of digits (0-9)
    count = new int[10]

    // Initialize an output array to store the sorted data
    output = new int[numElements]

    // Perform counting sort for each digit, from the least significant to the most significant
    for digitPosition from 1 to numDigits:
        // Reset the count array
        for i from 0 to 9:
            count[i] = 0

        // Count the occurrences of each digit in data
        for i from 0 to numElements:
            digit = getDigit(data[i], digitPosition)
            count[digit]++

        // Update the count array to contain the actual position of each digit in the output
        for i from 1 to 9:
            count[i] += count[i - 1]

        // Build the output array using the count array
        for i from numElements - 1 down to 0:
            digit = getDigit(data[i], digitPosition)
            output[count[digit] - 1] = data[i]
            count[digit]--

        // Copy the output array back to the data array
        for i from 0 to numElements:
            data[i] = output[i]

    // Free memory
    free(count)
    free(output)

Radix Sort (MPI+CUDA): ****************************************************************************

function ParallelRadixSort(data, numElements):
   1. Get the rank and number of processes
   2. Calculate the size of local data
   3. Allocate memory for local data
   4. Distribute data among processes
   5. Initialize CUDA for the current process
   6. Perform one pass for each bit (for loop)
      a. Sort local data using CUDA
      b. Synchronize CUDA threads
      c. Exchange data among processes
   7. Use a for loop to output sorted data on each process
   8. Cleanup by freeing memory

end function

Quicksort (MPI) ****************************************************************************
https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/#

function quicksort(arr, start, end)
// declare pivot and index
// consider base case if arr has only one element
// swap pivot with first element (pivot is middle element)
// set index to start of the arr
// loop from start+1 to start+end
    // swap if arr[i] < pivot
// swap pivot into element at index
// recursively call sorting function from both sides of the list


Quicksort (CUDA)
// define partitioning function
// __device__ int partition(int* arr, int left, int right)
// choose right element as pivot
// calculate index of left element
// loop from left to right j=left->right
    // swap arr[i] and arr[j] if arr[j] < pivot
// swap pivot element with element at i+1
// return index of pivot element


// define cudaQuicksort function
// void cudaQuicksort(int* arr, int size)
// declare device array
// launch quicksort kernel on device array
// launch quicksort kernel on device
// cudaMemcpy(arr, device array,..,cudaMemcyDeviceToHost)
// cudaFree(device array)
