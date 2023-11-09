#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


// Sources: geeksforgeeks.org/merge-sort, https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
int* mergeSort(int height, int id, int localArray[], int size, MPI_Comm comm, int globalArray[]){
    int parent, rightChild, myHeight;
    int *half1, *half2, *mergeResult;

    myHeight = 0;
    qsort(localArray, size, sizeof(int), compare); // sort local array
    half1 = localArray;  // assign half1 to localArray
	
    while (myHeight < height) { // not yet at top
        parent = (id & (~(1 << myHeight)));

        if (parent == id) { // left child
		    rightChild = (id | (1 << myHeight));

  		    // allocate memory and receive array of right child
  		    half2 = (int*) malloc (size * sizeof(int));
  		    MPI_Recv(half2, size, MPI_INT, rightChild, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  		    // allocate memory for result of merge
  		    mergeResult = (int*) malloc (size * 2 * sizeof(int));
  		    // merge half1 and half2 into mergeResult
  		    mergeResult = merge(half1, half2, mergeResult, size);
  		    // reassign half1 to merge result
            half1 = mergeResult;
			size = size * 2;  // double size
			
			free(half2); 
			mergeResult = NULL;

            myHeight++;

        } else { // right child
			  // send local array to parent
              MPI_Send(half1, size, MPI_INT, parent, 0, MPI_COMM_WORLD);
              if(myHeight != 0) free(half1);  
              myHeight = height;
        }
    }

    if(id == 0){
		globalArray = half1;   // reassign globalArray to half1
	}
	return globalArray;
}


int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;

    int numProcs, id, globalArraySize, localArraySize, height;
    int *localArray, *globalArray;
    double startTime, localTime, totalTime;
    double zeroStartTime, zeroTotalTime, processStartTime, processTotalTime;;
    int length = -1;
    char myHostName[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Get_processor_name (myHostName, &length); 

    // check for odd processes
    powerOfTwo(id, numProcs);

    // get size of global array
    getInput(argc, argv, id, numProcs, &globalArraySize);

    // calculate total height of tree
    height = log2(numProcs);

    // if process 0, allocate memory for global array and fill with values
    if (id==0){
		globalArray = (int*) malloc (globalArraySize * sizeof(int));
		fillArray(globalArray, globalArraySize, id);
		//printList(id, "UNSORTED ARRAY", globalArray, globalArraySize);  // Line A
	}
	
    // allocate memory for local array, scatter to fill with values and print
    localArraySize = globalArraySize / numProcs;
    localArray = (int*) malloc (localArraySize * sizeof(int));
    MPI_Scatter(globalArray, localArraySize, MPI_INT, localArray, 
		localArraySize, MPI_INT, 0, MPI_COMM_WORLD);
    //printList(id, "localArray", localArray, localArraySize);   // Line B 
    
    //Start timing
    startTime = MPI_Wtime();
    //Merge sort
    if (id == 0) {
		zeroStartTime = MPI_Wtime();
		globalArray = mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, globalArray);
		zeroTotalTime = MPI_Wtime() - zeroStartTime;
		printf("Process #%d of %d on %s took %f seconds \n", 
			id, numProcs, myHostName, zeroTotalTime);
	}
	else {
		processStartTime = MPI_Wtime();
	        mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, NULL);
		processTotalTime = MPI_Wtime() - processStartTime;
		printf("Process #%d of %d on %s took %f seconds \n", 
			id, numProcs, myHostName, processTotalTime);
	}
    //End timing
    localTime = MPI_Wtime() - startTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE,
        MPI_MAX, 0, MPI_COMM_WORLD);

    if (id == 0) {
		//printList(0, "FINAL SORTED ARRAY", globalArray, globalArraySize);  // Line C
		printf("Sorting %d integers took %f seconds \n", globalArraySize,totalTime);
		free(globalArray);
	}

    free(localArray);  


    cali::ConfigManager mgr;
	mgr.start();

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "integer");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("num_procs", num_procs);
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "Online"); 


    mgr.stop();
   	mgr.flush();

    MPI_Finalize();
    return 0;

}