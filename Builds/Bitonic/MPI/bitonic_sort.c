#include <stdio.h>      // Printf
#include <time.h>       // Timer
#include <math.h>       // Logarithm
#include <stdlib.h>     // Malloc
#include "mpi.h"        // MPI Library
#include <string.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0        

// SOURCE: https://github.com/sajid912/MPI-Bitonic-Sort/blob/master/bitonic_sort.c

// Globals
double timer_start;
double timer_end;
int process_rank;
int num_processes;
int *array;
int array_size;

unsigned int Log2n(unsigned int n) {
    return (n > 1) ? 1 + Log2n(n/2) : 0;
}

int main(int argc, char * argv[]) {

    cali::ConfigManager mgr;
    mgr.start();
    CALI_CXX_MARK_FUNCTION("main");

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    array_size = atoi(argv[1]) / num_processes;
    array = (int *) malloc(array_size * sizeof(int));

    srand(time(NULL) + process_rank * num_processes);  
    for (int i = 0; i < array_size; i++) {
        array[i] = rand() % (atoi(argv[1]));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int dimensions = Log2n(num_processes);

    if (process_rank == MASTER) {
        printf("Number of Processes spawned: %d\n", num_processes);
        CALI_CXX_MARK_BEGIN("overall_sort_timer");
        timer_start = MPI_Wtime();
    }

    CALI_CXX_MARK_BEGIN("initial_sort");
    qsort(array, array_size, sizeof(int), ComparisonFunc);
    CALI_CXX_MARK_END("initial_sort");

    CALI_CXX_MARK_BEGIN("bitonic_sort_phases");
    for (int i = 0; i < dimensions; i++) {
        for (int j = i; j >= 0; j--) {
            CALI_CXX_MARK_BEGIN("bitonic_sort_phase");
            if (((process_rank >> (i + 1)) % 2 == 0 && (process_rank >> j) % 2 == 0) ||
                ((process_rank >> (i + 1)) % 2 != 0 && (process_rank >> j) % 2 != 0)) {
                CompareLow(j);
            } else {
                CompareHigh(j);
            }
            CALI_CXX_MARK_END("bitonic_sort_phase");
        }
    }
    CALI_CXX_MARK_END("bitonic_sort_phases");

    MPI_Barrier(MPI_COMM_WORLD);

    if (process_rank == MASTER) {
        timer_end = MPI_Wtime();
        CALI_CXX_MARK_END("overall_sort_timer");
        printf("Time Elapsed (Sec): %f\n", timer_end - timer_start);
    }

     // Report values using Adiak
    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "BitonicSort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_processes); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS);
    // adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", "4");
    adiak::value("implementation_source", "Online"); 
    // Finalize Adiak
    // adiak::fini();

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    free(array);
    MPI_Finalize();
    return 0;
}

int ComparisonFunc(const void * a, const void * b) {
    return (*(int *)a - *(int *)b);
}

void CompareLow(int j) {
    CALI_CXX_MARK_FUNCTION("compare_low");
    // ... existing code for CompareLow ...
    int i, min;

   // printf("My rank is %d Pairing with %d in CL\n", process_rank, process_rank^(1<<j));
    
    int send_counter = 0;
    int *buffer_send = malloc((array_size + 1) * sizeof(int));
   // printf("Trying to send local max in CL:%d\n", array[array_size-1]);
    MPI_Send(
        &array[array_size - 1],     
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );

    int recv_counter;
    int *buffer_recieve = malloc((array_size + 1) * sizeof(int));
    MPI_Recv(
        &min,                       
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );
    
   // printf("Received min from pair in CL:%d\n", min);
    for (i = array_size-1; i >= 0; i--) {
        if (array[i] > min) {
	    send_counter++;
            buffer_send[send_counter] = array[i];
	   // printf("Buffer sending in CL %d\n", array[i]);
        
        } else {
             break;      
        }
    }

    buffer_send[0] = send_counter;
   // printf("Send count in CL: %d\n", send_counter);

//	for(i=0;i<=send_counter;i++)
//		printf(" %d?? ", buffer_send[i]);
    MPI_Send(
        buffer_send,                
        send_counter+1,               
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );

    MPI_Recv(
        buffer_recieve,             
        array_size+1,                 
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );

    int *temp_array = (int *) malloc(array_size * sizeof(int));
    //memcpy(temp_array, array, array_size * sizeof(int));
    for(i=0;i<array_size;i++)
	temp_array[i]=array[i];
   
    int buffer_size=buffer_recieve[0];
    int k=1;int m=0;   

  //  for(k=0;k<=buffer_size;k++)
//	printf(" %d? ", buffer_recieve[k]);
 
    k=1;
    for (i = 0; i < array_size; i++) {
	//printf("Receive buffer element in CL: %d\n", buffer_recieve[i]);
    //    if (array[array_size - 1] < buffer_recieve[i]) {
      //      array[array_size - 1] = buffer_recieve[i];
       // } else {
         //   break;      
        //}
	if(temp_array[m]<=buffer_recieve[k])
	{
		array[i]=temp_array[m];
		m++;
	}
        else if(k<=buffer_size)
	{
		array[i]=buffer_recieve[k];
		k++;
	}
    }

    qsort(array, array_size, sizeof(int), ComparisonFunc);
     //  for(i=0;i<array_size;i++)
    //	printf("My rank is %d, after exchange in CL %d\n", process_rank, array[i]);

    // int s=0;
    //for(i=0;i<array_size;i++)
        
    //	{
    //		printf("%d", array[i]);	
    //		for(s=0;s<=process_rank;s++)
    //			printf(":");
    //	}
        
    //	printf("\n");
	
    free(buffer_send);
    free(buffer_recieve);

    return;
}

void CompareHigh(int j) {
    CALI_CXX_MARK_FUNCTION("compare_high");
    //printf("My rank is %d Pairing with %d in CH\n", process_rank, process_rank^(1<<j));
    int i, max;

    int recv_counter;
    int *buffer_recieve = malloc((array_size + 1) * sizeof(int));
    MPI_Recv(
        &max,                       
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );

   // printf("Received max from pair in CH:%d\n",max);
    int send_counter = 0;
    int *buffer_send = malloc((array_size + 1) * sizeof(int));
    MPI_Send(
        &array[0],                  
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );

   // printf("Sending min to my pair from CH:%d\n", array[0]);
    for (i = 0; i < array_size; i++) {
        if (array[i] < max) {
	   // printf("Buffer sending in CH: %d\n", array[i]);
            	send_counter++;
		buffer_send[send_counter] = array[i];
        } else {
            break;      
        }
    }

    
    MPI_Recv(
        buffer_recieve,             
        array_size+1,                 
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );
    recv_counter = buffer_recieve[0];

    buffer_send[0] = send_counter;
    //printf("Send counter in CH: %d\n", send_counter);

  //  for(i=0;i<=send_counter;i++)
//	printf(" %d>> ", buffer_send[i]);
    MPI_Send(
        buffer_send,               
        send_counter+1,               
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );
    int *temp_array = (int *) malloc(array_size * sizeof(int));
    //memcpy(temp_array, array, array_size * sizeof(int));
    for(i=0;i<array_size;i++)
	temp_array[i]=array[i];

    int k=1;int m=array_size-1;
    int buffer_size=buffer_recieve[0];
		
    //for(i=0;i<=buffer_size;i++)
//	printf(" %d> ", buffer_recieve[i]);
    for (i = array_size-1; i >= 0; i--) {
	//printf("Buffer receive ele in CH: %d\n", buffer_recieve[i]);
        //if (buffer_recieve[i] > array[0]) {
          //  array[0] = buffer_recieve[i];
        //} else {
          //  break;      
        //}
      //  printf("buffer_rec[k] is %d, temp_array[m] is %d\n",buffer_recieve[k], temp_array[m]);
//	printf("M is %d k is %d i is %d\n",m, k, i);
	if(temp_array[m]>=buffer_recieve[k])
	{
		array[i]=temp_array[m];
		m--;
	}
	else if(k<=buffer_size){
		array[i]=buffer_recieve[k];
		k++;
	}
    }

    qsort(array, array_size, sizeof(int), ComparisonFunc);
	
   //int s=0;
   //for(i=0;i<array_size;i++)
	
	//{
	//	printf("%d", array[i]);	
	//	for(s=0;s<=process_rank;s++)
	//		printf(",");
	//}

    //printf("\n");
    free(buffer_send);
    free(buffer_recieve);

    return;
    
}
