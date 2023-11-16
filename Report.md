# CSCE 435 Group project

## 1. Group members:
1. Emma Ong
2. Naimur Rahman
3. Anna Huang
4. Luis Martinez Morales

We will be communicating through Discord and text messaging.

---

## 2. Project topic: Comparative Analysis of Sorting Algorithms Using MPI and CUDA

## 3. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
Our project aims to compare the performance of various parallel sorting algorithms using two different parallel computing technologies: Message Passing Interface (MPI) and Compute Unified Device Architecture (CUDA). We will be comparing Merge Sort, Quick Sort, Radix Sort, & Bitonic Sort. For Merge Sort and Quick Sort we will be comparing the MPI implementation versus CUDA implementation. For Radix Sort we will be comparing the MPI implementation to the CUDA implementation. For Bitonic Sort we will be comparing the implementation to the CUDA implementation.


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

Radix Sort (MPI): ****************************************************************************

Initialize MPI environment.

Each process receives a chunk of the unsorted array.

In parallel, each process performs Radix Sort on its local chunk.

For each digit (starting from the least significant digit):
    Each process counts the occurrences of each digit within its chunk.
    All processes participate in a collective communication to determine the global count for each    digit.
    Each process determines the target process for each element based on the global digit counts and sends the elements to the respective processes.

After all digits have been processed, each process has a portion of the globally sorted array.

If necessary, merge the sorted chunks from each process to form the fully sorted array.

Finalize MPI environment.


Quicksort (MPI)
https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/#

function quicksort(arr, start, end)
1. declare pivot and index
2. consider base case if arr has only one element
3. swap pivot with first element (pivot is middle element)
4. set index to start of the arr
5. loop from start+1 to start+end
    a. swap if arr[i] < pivot
6. swap pivot into element at index
7. recursively call sorting function from both sides of the list


Quicksort (CUDA)
1. define partitioning function
2. __device__ int partition(int* arr, int left, int right)
3. choose right element as pivot
4. calculate index of left element
5. loop from left to right j=left->right
    a. swap arr[i] and arr[j] if arr[j] < pivot
6. return index of pivot element


7. define cudaQuicksort function
8. void cudaQuicksort(int* arr, int size)
9. declare device array
10. launch quicksort kernel on device array
11. launch quicksort kernel on device
12. cudaMemcpy(arr, device array,..,cudaMemcyDeviceToHost)
13. cudaFree(device array)

**************************************************************************************************
Bitonic Sort (MPI)
function bitonicSort(up, sequence)
  if length(sequence) > 1 then
    firstHalf = first half of sequence
    secondHalf = second half of sequence

    bitonicSort(true, firstHalf)  // sort in ascending order
    bitonicSort(false, secondHalf) // sort in descending order

    bitonicMerge(up, sequence) // merge whole sequence in ascending or descending order

function bitonicMerge(up, sequence)
  if length(sequence) > 1 then
    // bitonic split
    compareAndSwap(up, sequence)

    firstHalf = first half of sequence
    secondHalf = second half of sequence

    bitonicMerge(up, firstHalf)
    bitonicMerge(up, secondHalf)

function compareAndSwap(up, sequence)
  distance = length(sequence) / 2
  for i = 0 to distance - 1 do
    if (up and sequence[i] > sequence[i + distance]) or (!up and sequence[i] < sequence[i + distance]) then
      swap(sequence[i], sequence[i + distance])

// To sort a sequence in ascending order using bitonic sort:
bitonicSort(true, sequence)

**************************************************************************************************
Bitonic Sort (CUDA)

Set number of threads, blocks, and number of values to sort

Allocate memory for values on the host
Fill the host memory with random floating-point numbers

Allocate memory on the device (GPU) for sorting

Copy the unsorted values from host to device memory

For each stage of the bitonic sequence:
  For each step of the current stage:
    Launch the bitonic sort kernel with the current step and stage parameters
    The kernel will compare and swap elements to achieve the bitonic sequence

Wait for GPU to finish sorting

Copy the sorted values from device back to host memory

Free the device memory

Print the sorted values (if needed)




## 2c. Evaluation plan - what and how you will measure and compare
Regarding input types, one will be an integer value for the size of the array that holds the values to be sorted and the other is the thread count used for the algorithm. Array sizes will be {16, 64, 128, 256, 1024, 2048, 4096} and thread sizes will be  {2, 4, 16, 32, 64, 128, 256, 512}. Each array size will be tested with every thread size. All arrays will be filled with integer values as radix sort is only possible with integers and not floating point numbers. The values for arrays will be randomly generated within the program depending on problem size.
For strong scaling, each problem size will be tested with the aforementioned increasing amount of thread sizes. For weak scaling, all thread counts will be tested with increasing array problem sizes for sorting. 
Overall, we will be testing and comparing overall run times with all four algorithms and their subset types. The run times will be compared based on the factors of thread count, algorithm, and problem size. 


## 3c. Project Implementation 
Originally, our plan involved identifying and utilizing sources to implement our algorithms, followed by compiling and executing our code on the Grace platform. We had meticulously conducted our research and prepared pseudocode well in advance. However, as soon as we transitioned from pseudocode to actual code that we could run, Grace underwent an extended maintenance period, preventing us from testing our code and generating calibration files.
Hypothesis: 
We estimate that quicksort would have a lower runtime than merge, radix, and bitonic sort because of their overall scaled predicted run times. When comparing, the runtime for parallel implementation of Quick sort (theoretical best) is O(logn/p) where p is the number of parallel processors, but the worst case is O(n^2). The runtime for parallel implementation of Merge sort using cuda is O(nlog(n/p)) where p is the number of processors with a good implementation of parallelized merge sort.
The time complexity of Radix Sort is O(w*n) for the serial/sequential version, where n is the number of elements needed to be sorted and w is the number of bits required to store each key. Radix Sort works by processing each bit of the numbers to be sorted, which leads to its linear time complexity in the number of bits processed. However, in a parallel CUDA implementation, Radix Sort's time complexity can be significantly reduced by distributing the counting and prefix sum computation across the multiple cores of the GPU. Theoretically, if you have as many processing units as items to be sorted, the time complexity could approach O(w), since you could potentially process each bit of all items simultaneously.
The Bitonic Sort algorithm has an O(logn) time complexity. However when it comes to the parallelized algorithm implementation with CUDA theoretically should be O(log (n/p)) where p is the number of processors. 
In conclusion, based on these analytical time complexities when parallelized, it is clear to see why quicksort would be the most optimal. Parallelizing these algorithms as a whole will greatly reduce run times of the algorithms.

## 4. Performance evaluation

![Bitonic Sort CUDA Graph](Graphs/bitonic_cuda.png "Bitonic Sort CUDA Graph")
