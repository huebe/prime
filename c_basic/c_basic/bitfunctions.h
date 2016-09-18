//
// Created by ch1pmunk on 18.09.16.
//

#ifndef BITFUNCTIONS_H
#define BITFUNCTIONS_H


// unsigned long long helper macros
#define ULONGLONG_SIZE ( sizeof(unsigned long long))
#define ULONGLONG_SIZE_BITS ( ULONGLONG_SIZE * 8)

#define GET_BITMASK_ULONGLONG(i) ( (unsigned int)1 << (i % (ULONGLONG_SIZE_BITS)) )
#define GET_ARRAY_ULONGLONG(i) ( i/ (ULONGLONG_SIZE))
#define SET_BIT_ULONGLONG_ARRAY(arr, i) ( arr[i/ULONGLONG_SIZE_BITS] |= 1ULL << (i % ULONGLONG_SIZE_BITS) )
#define CLEAR_BIT_ULONGLONG_ARRAY(arr, i) ( arr[i/ULONGLONG_SIZE_BITS] &= ~(char)(1ULL << (i % ULONGLONG_SIZE_BITS)) )


// int helper macros
#define INT_SIZE ( sizeof(int) )
#define INT_SIZE_BITS ( INT_SIZE * 8)

#define GET_BITMASK_INT(i) ( (unsigned int)1 << (i % (INT_SIZE_BITS)) )
#define GET_ARRAY_ELEMENT(i) ( i/ (INT_SIZE_BITS))
#define SET_BIT_INT_ARRAY(arr, i) ( arr[i/INT_SIZE_BITS] |= 1 << (i % INT_SIZE_BITS) )
#define CLEAR_BIT_INT_ARRAY(arr, i) ( arr[i/INT_SIZE_BITS] &= ~(char)(1 << (i % INT_SIZE_BITS)) )
#define READ_BIT_INT_ARRAY(arr, i) ( arr[i/INT_SIZE_BITS] & 1 << (i % INT_SIZE_BITS) )

#endif //BITFUNCTIONS_H
