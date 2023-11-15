// #include <iostream>
// #include <cstdlib>
// #include <cstring>
// #include <cmath>
// #include <mpi.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <limits.h>
// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>


// // Sources: geeksforgeeks.org/merge-sort, https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
// int* mergeSort(int height, int id, int localArray[], int size, MPI_Comm comm, int globalArray[]){
//     int parent, rightChild, myHeight;
//     int *half1, *half2, *mergeResult;

//     myHeight = 0;
//     CALI_MARK_BEGIN(comp);
// 	CALI_MARK_BEGIN(compLarge);
//     qsort(localArray, size, sizeof(int), compare); // sort local array
//     CALI_MARK_END(compLarge);
// 	CALI_MARK_END(comp);
//     half1 = localArray;  // assign half1 to localArray
	
//     while (myHeight < height) { // not yet at top
//         parent = (id & (~(1 << myHeight)));

//         if (parent == id) { // left child
// 		    rightChild = (id | (1 << myHeight));

//   		    // allocate memory and receive array of right child
//   		    half2 = (int*) malloc (size * sizeof(int));
//             CALI_MARK_BEGIN(comm);
// 			CALI_MARK_BEGIN(commLarge);
//   		    MPI_Recv(half2, size, MPI_INT, rightChild, 0,
// 				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             CALI_MARK_END(commLarge);
// 			CALI_MARK_END(comm);

//   		    // allocate memory for result of merge
//   		    mergeResult = (int*) malloc (size * 2 * sizeof(int));
//   		    // merge half1 and half2 into mergeResult
//             CALI_MARK_BEGIN(comp);
// 			CALI_MARK_BEGIN(compLarge);
//   		    mergeResult = merge(half1, half2, mergeResult, size);
//             CALI_MARK_END(compLarge);
// 			CALI_MARK_END(comp);
//   		    // reassign half1 to merge result
//             half1 = mergeResult;
// 			size = size * 2;  // double size
			
// 			free(half2); 
// 			mergeResult = NULL;

//             myHeight++;

//         } else { // right child
// 			  // send local array to parent
//               CALI_MARK_BEGIN(comm);
//               CALI_MARK_BEGIN(commLarge);
//               MPI_Send(half1, size, MPI_INT, parent, 0, MPI_COMM_WORLD);
//               CALI_MARK_END(commLarge);
//               CALI_MARK_END(comm);
//               if(myHeight != 0) free(half1);  
//               myHeight = height;
//         }
//     }

//     if(id == 0){
// 		globalArray = half1;   // reassign globalArray to half1
// 	}
// 	return globalArray;
// }


// int main(int argc, char** argv) {
//     CALI_CXX_MARK_FUNCTION;
//     CALI_MARK_BEGIN(main);

//     int numProcs, id, globalArraySize, localArraySize, height;
//     int *localArray, *globalArray;
//     double startTime, localTime, totalTime;
//     double zeroStartTime, zeroTotalTime, processStartTime, processTotalTime;;
//     int length = -1;
//     char myHostName[MPI_MAX_PROCESSOR_NAME];

//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
//     MPI_Comm_rank(MPI_COMM_WORLD, &id);

//     MPI_Get_processor_name (myHostName, &length); 

//     // check for odd processes
//     powerOfTwo(id, numProcs);

//     // get size of global array
//     getInput(argc, argv, id, numProcs, &globalArraySize);

//     // calculate total height of tree
//     height = log2(numProcs);

//     // if process 0, allocate memory for global array and fill with values
//     if (id==0){
// 		globalArray = (int*) malloc (globalArraySize * sizeof(int));
// 		fillArray(globalArray, globalArraySize, id);
// 		//printList(id, "UNSORTED ARRAY", globalArray, globalArraySize);  // Line A
// 	}
	
//     // allocate memory for local array, scatter to fill with values and print
//     localArraySize = globalArraySize / numProcs;
//     localArray = (int*) malloc (localArraySize * sizeof(int));
//     MPI_Scatter(globalArray, localArraySize, MPI_INT, localArray, 
// 		localArraySize, MPI_INT, 0, MPI_COMM_WORLD);
//     //printList(id, "localArray", localArray, localArraySize);   // Line B 
    
//     //Start timing
//     startTime = MPI_Wtime();
//     //Merge sort
//     if (id == 0) {
// 		zeroStartTime = MPI_Wtime();
//         CALI_MARK_BEGIN(comp);
// 		CALI_MARK_BEGIN(compLarge);
// 		globalArray = mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, globalArray);
//         CALI_MARK_END(compLarge);
// 		CALI_MARK_END(comp);
// 		zeroTotalTime = MPI_Wtime() - zeroStartTime;
// 		printf("Process #%d of %d on %s took %f seconds \n", 
// 			id, numProcs, myHostName, zeroTotalTime);
// 	}
// 	else {
// 		processStartTime = MPI_Wtime();
// 	        mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, NULL);
// 		processTotalTime = MPI_Wtime() - processStartTime;
// 		printf("Process #%d of %d on %s took %f seconds \n", 
// 			id, numProcs, myHostName, processTotalTime);
// 	}
//     //End timing
//     localTime = MPI_Wtime() - startTime;
//     MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE,
//         MPI_MAX, 0, MPI_COMM_WORLD);

//     if (id == 0) {
// 		//printList(0, "FINAL SORTED ARRAY", globalArray, globalArraySize);  // Line C
// 		printf("Sorting %d integers took %f seconds \n", globalArraySize,totalTime);
// 		free(globalArray);
// 	}

//     free(localArray);  
//     CALI_MARK_END(main);


//     cali::ConfigManager mgr;
// 	mgr.start();

//     adiak::init(NULL);
//     adiak::user();
//     adiak::launchdate();
//     adiak::libraries();
//     adiak::cmdline();
//     adiak::clustername();
//     adiak::value("Algorithm", "Merge Sort");
//     adiak::value("ProgrammingModel", "MPI");
//     adiak::value("Datatype", "int");
//     adiak::value("SizeOfDatatype", sizeof(int));
//     adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
//     adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", etc.
//     adiak::value("num_procs", 2);
//     adiak::value("num_threads", THREADS);
//     adiak::value("num_blocks", BLOCKS);
//     adiak::value("num_vals", NUM_VALS);
//     adiak::value("group_num", "4");
//     adiak::value("implementation_source", "Online"); 


//     mgr.stop();
//    	mgr.flush();

//     MPI_Finalize();
//     return 0;

// }

// Source Code from: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c by racorretjer (Roberto Arce Corretjer)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";
const char *correctness_check = "correctness_check";

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

int main(int argc, char** argv) {
	CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
	/********** Create and populate the array **********/
	int n = atoi(argv[1]);
	int *original_array = malloc(n * sizeof(int));
	
	int c;
	srand(time(NULL));
	printf("This is the unsorted array: ");

    CALI_MARK_BEGIN(data_init);
	for(c = 0; c < n; c++) {
		
		original_array[c] = rand() % n;
		printf("%d ", original_array[c]);
		
		}
    CALI_MARK_END(data_init);

	printf("\n");
	printf("\n");
	
	/********** Initialize MPI **********/
	int world_rank;
	int world_size;
	
	MPI_INIT(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		
	/********** Divide the array in equal-sized chunks **********/
	int size = n/world_size;
	
	/********** Send each subarray to each process **********/
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    int *sub_array = malloc(size * sizeof(int));
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

	/********** Perform the mergesort on each process **********/
	CALI_MARK_BEGIN(comp);
	CALI_MARK_BEGIN(comp_large);
    int *tmp_array = malloc(size * sizeof(int));
	mergeSort(sub_array, tmp_array, 0, (size - 1));
    CALI_MARK_END(comp_large);
	CALI_MARK_END(comp);
	
	/********** Gather the sorted subarrays into one **********/
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    int *sorted = NULL;
	if(world_rank == 0) {
		
		sorted = malloc(n * sizeof(int));
		
		}
	
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

	/********** Make the final mergeSort call **********/
	if(world_rank == 0) {
		
		int *other_array = malloc(n * sizeof(int));
		mergeSort(sorted, other_array, 0, (n - 1));
		
		/********** Display the sorted array **********/
		printf("This is the sorted array: ");
		for(c = 0; c < n; c++) {
			
			printf("%d ", sorted[c]);
			
			}
			
		printf("\n");
		printf("\n");

        if(isSorted(sorted, n)) {
            printf("\n");
		    printf("\n");
            printf("Array is sorted!");
            printf("\n");
		    printf("\n");
        } else {
            printf("\n");
		    printf("\n");
            printf("Array is not sorted!");
            printf("\n");
		    printf("\n");
        }
			
		/********** Clean up root **********/
		free(sorted);
		free(other_array);
			
		}
	
	/********** Clean up rest **********/
	free(original_array);
	free(sub_array);
	free(tmp_array);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", etc.
    adiak::value("num_procs", world_size);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "Online"); 

	
	/********** Finalize MPI **********/
	MPI_Barrier(MPI_COMM_WORLD);

    mgr.stop();
    mgr.flush();
	MPI_Finalize();
	
	}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r) {
	CALI_MARK_BEGIN(comp_small);
	int h, i, j, k;
	h = l;
	i = l;
	j = m + 1;
	
	while((h <= m) && (j <= r)) {
		
		if(a[h] <= a[j]) {
			
			b[i] = a[h];
			h++;
			
			}
			
		else {
			
			b[i] = a[j];
			j++;
			
			}
			
		i++;
		
		}
		
	if(m < h) {
		
		for(k = j; k <= r; k++) {
			
			b[i] = a[k];
			i++;
			
			}
			
		}
		
	else {
		
		for(k = h; k <= m; k++) {
			
			b[i] = a[k];
			i++;
			
			}
			
		}
		
	for(k = l; k <= r; k++) {
		
		a[k] = b[k];
		
		}
	CALI_MARK_END(comp_small);	
	}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r) {
	CALI_MARK_BEGIN(comp_large);
	int m;
	
	if(l < r) {
		
		m = (l + r)/2;
		
		mergeSort(a, b, l, m);
		mergeSort(a, b, (m + 1), r);
		merge(a, b, l, m, r);
		
		}
	CALI_MARK_END(comp_large);	
	}


bool isSorted(const int *array, int size) {
    CALI_MARK_BEGIN(correctness_check);
    if (size < 2) {
        return true; // An array with 0 or 1 element is always sorted
    }

    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            CALI_MARK_END(correctness_check);
            return false; // Found a pair of elements that are out of order
        }
    }

    CALI_MARK_END(correctness_check);
    return true; // No elements are out of order
}