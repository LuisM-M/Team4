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


