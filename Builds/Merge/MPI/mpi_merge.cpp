#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

const char* correctness_check = "correctness_check";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* MPI_gather = "MPI_Gather";
const char* MPI_scatter = "MPI_Scatter";
const char* MPI_barrier = "MPI_Barrier";




void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

bool isSorted(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
	CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
	/********** Create and populate the array **********/
	int n = atoi(argv[1]);
	int *original_array = (int*)malloc(n * sizeof(int));
	
	CALI_MARK_BEGIN(data_init);
	int c;
	srand(time(NULL));
	for (c = 0; c < n; c++) {
    	original_array[c] = rand() % n;
	}
	CALI_MARK_END(data_init);
	/********** Initialize MPI **********/
	int world_rank;
	int world_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		
	/********** Divide the array into equal-sized chunks **********/
	int size = n/world_size;
	
	/********** Send each subarray to each process **********/
	int *sub_array = (int*)malloc(size * sizeof(int));
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_scatter);
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(MPI_scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
	
	/********** Perform the mergesort on each process **********/
	CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
	int *tmp_array = (int*)malloc(size * sizeof(int));
	mergeSort(sub_array, tmp_array, 0, (size - 1));
	CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
	
	/********** Gather the sorted subarrays into one **********/
	int *sorted = NULL;
	if(world_rank == 0) {
		
		sorted = (int*)malloc(n * sizeof(int));
		
	}
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_gather);
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(MPI_gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
	
	/********** Make the final mergeSort call **********/
	if(world_rank == 0) {
		
		int *other_array = (int*)malloc(n * sizeof(int));
		mergeSort(sorted, other_array, 0, (n - 1));
			
		CALI_MARK_BEGIN(correctness_check);
		if (isSorted(sorted, n)) {
			printf("The array is correctly sorted.\n");
		} else {
			printf("The array is NOT correctly sorted.\n");
		}
		CALI_MARK_END(correctness_check);
		/********** Clean up root **********/
		free(sorted);
		free(other_array);
	}

	/********** Clean up rest **********/
	free(original_array);
	free(sub_array);
	free(tmp_array);

	/********** Finalize MPI **********/
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPI_barrier);
	MPI_Barrier(MPI_COMM_WORLD);
	CALI_MARK_END(MPI_barrier);
    CALI_MARK_END(comm);
	mgr.stop();
    mgr.flush();

    MPI_Finalize();
}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r) {
	
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
		
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r) {
	
	int m;
	
	if(l < r) {
		
		m = (l + r)/2;
		
		mergeSort(a, b, l, m);
		mergeSort(a, b, (m + 1), r);
		merge(a, b, l, m, r);
		
	}
		
}
