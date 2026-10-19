// Make sure our function (symbols) are visible in the produced shared library (DLL / .so)
#ifdef _WIN32
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif
// Also use extern "C" so that the names do not get mangled.

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>


/* 
C wrapper for the actual sorting in cub's SortPairs, 
instantiated for 64-bit integer (long) keys, 32-bit int integer (int) values.
(We tested this to be slightly faster than using the unsigned variants.)

Arguments:
    d_temp_storage:         A memory buffer (void*).
    temp_storage_bytes:     The (required) size of d_temp_storage (size_t).
                                You can use determine_sort_pairs_temp_storage for this.
    d_keys_in/out:          The array to sort (int64_t*). 
                                Input (will not change, const) / output (will contain the sorted version).
    d_values_in_out:        Will be cosorted (int32_t*) via the keys. Similar input/output structure.
    num_items:              The length of the (part of the) array to sort (int32_t).
    highest_key_bit:        The index of the most significant bit in keys needed for the sort (int32_t). 
                                0-indexed, exclusive. E.g. 64 if we need all bits. 
                                (We choose to always start at the least significant bit.)
*/
extern "C" EXPORT_SYMBOL void sort_pairs(
    void* d_temp_storage, size_t temp_storage_bytes, 
    const std::int64_t* d_keys_in, std::int64_t* d_keys_out, 
    const std::int32_t* d_values_in, std::int32_t* d_values_out, 
    int num_items, int highest_key_bit) 
{
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, 
        d_keys_in, d_keys_out, 
        d_values_in, d_values_out, 
        num_items, 
        0, highest_key_bit);
}

/*
Determine the size in bytes of the temporary storage we will need for the actual SortPairs call.
Does not mutate any array.
*/
extern "C" EXPORT_SYMBOL size_t determine_sort_pairs_temp_storage(
    const std::int64_t* d_keys_in, std::int64_t* d_keys_out, 
    const std::int32_t* d_values_in, std::int32_t* d_values_out, 
    int num_items, int highest_key_bit)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, // pass by reference, will be updated
        d_keys_in, d_keys_out, 
        d_values_in, d_values_out, 
        num_items, 
        0, highest_key_bit);
    return temp_storage_bytes;
}