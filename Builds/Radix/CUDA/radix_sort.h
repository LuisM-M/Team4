// Author: Jack Liu
// Username: jackfly
// Source: https://github.com/jackfly/radix-sort-cuda

#ifndef RADIX_SORT_H__
#define RADIX_SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"
#include <cmath>

void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len,
    unsigned int num_threads);

#endif