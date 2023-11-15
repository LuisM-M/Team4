// Source Code from: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c by racorretjer (Roberto Arce Corretjer)
// I added the caliper & adiak implementations and a function to check that the array is sorted

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
bool isSorted(const int *array, int size);

int main(int argc, char** argv) {
	CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
	/********** Create and populate the array **********/
	int n = atoi(argv[1]);
	int *original_array = (int*)malloc(n * sizeof(int));
	
	int c;
	srand(time(NULL));
	// printf("This is the unsorted array: ");

    CALI_MARK_BEGIN(data_init);
	for(c = 0; c < n; c++) {
		
		original_array[c] = rand() % n;
		// printf("%d ", original_array[c]);
		
		}
    CALI_MARK_END(data_init);

	// printf("\n");
	// printf("\n");
	
	/********** Initialize MPI **********/
	int world_rank;
	int world_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		
	/********** Divide the array in equal-sized chunks **********/
	int size = n/world_size;
	
	/********** Send each subarray to each process **********/
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    int *sub_array = (int*)malloc(size * sizeof(int));
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
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
	CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    int *sorted = NULL;
	if(world_rank == 0) {
		
		sorted = (int*)malloc(n * sizeof(int));
		
		}
	
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
	CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

	/********** Make the final mergeSort call **********/
	if(world_rank == 0) {
		
		int *other_array = (int*)malloc(n * sizeof(int));
		mergeSort(sorted, other_array, 0, (n - 1));
		
		/********** Display the sorted array **********/
		// printf("This is the sorted array: ");
		// for(c = 0; c < n; c++) {
			
		// 	printf("%d ", sorted[c]);
			
		// 	}
			
		// printf("\n");
		// printf("\n");

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
    adiak::value("SizeOfDatatype", "4"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
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