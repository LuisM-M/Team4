#ifndef BITONIC_H
#define BITONIC_H

// Declares a function that performs a sequential sort on the data.
void sequentialSort(void);

// Declares a function that performs the "CompareLow" operation in the Bitonic sort.
void CompareLow(int bit);

// Declares a function that performs the "CompareHigh" operation in the Bitonic sort.
void CompareHigh(int bit);

// Declares the comparison function for qsort.
int ComparisonFunc(const void * a, const void * b);

#endif // BITONIC_H
