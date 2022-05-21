def get_insert_position(arr, len_arr, k):
    # time complexity: O(n)
    for i in range(len_arr):
        if k == arr[i]:
            return i
        elif k < arr[i]:
            return i
    return len_arr


def get_insert_position2(arr, arr_len, k):
    # Binary search-based. Time complexity: O(n)
    start_idx = 0
    end_idx = arr_len - 1

    while start_idx <= end_idx:
        mid_idx = (start_idx + end_idx) // 2
        if arr[mid_idx] == k:
            return mid_idx
        elif arr[mid_idx] < k:
            start_idx = mid_idx + 1
        else:
            end_idx = mid_idx - 1
    return end_idx + 1


def find_first_last(arr, arr_len, k):
    # Naive approach. Time complexity: O(n)
    flag = 0
    k_first_idx = 0
    for l in range(arr_len):
        if arr[l] == k:
            flag = 1
            k_first_idx = l
            print('First occurrence of {:d} is {:d}'.format(k, k_first_idx))
            break

    k_last_idx = k_first_idx
    for l in range(k_first_idx+1, arr_len):
        if arr[l] == k:
            k_last_idx = l
    if flag == 1:
        print('Last occurrence of {:d} is {:d}'.format(k, k_last_idx))
    else:
        print("No such element present")

def find_first_occ(arr, arr_len, k, low, high):
    # Binary search. Time complexity: O(N)
    while high >= low:
        mid = low + (high - low) // 2
        if (mid == 0 or k > arr[mid-1]) and arr[mid] == k:
            return mid
        elif k > arr[mid]:
            return find_first_occ(arr, arr_len, k, mid+1, high)
        else:
            return find_first_occ(arr, arr_len, k, low, mid-1)
    return -1

def find_last_occ(arr, arr_len, k, low, high):
    # Binary search. Time complexity: O(N)
    while high >= low:
        mid = low + (high - low) // 2
        if (mid == arr_len - 1 or k < arr[mid+1]) and arr[mid] == k:
            return mid
        elif k < arr[mid]:
            return find_last_occ(arr, arr_len, k, low, mid-1)
        else:
            return find_last_occ(arr, arr_len, k, mid+1, high)
    return -1

def find_pivot_index(arr, low, high):
    # Binary search. Time complexity: O(logn)
    # base cases
    if high < low:
        return -1
    if high == low:
        return low

    mid = (low + high)//2
    if mid < high and (arr[mid+1] < arr[mid]):
        return mid
    if mid > low and (arr[mid] < arr[mid-1]):
        return mid - 1
    if arr[low] >= arr[mid]:
        return find_pivot_index(arr, low, mid-1)
    return find_pivot_index(arr, mid+1, high)

def find_key_index(arr, low, high, key):
    #Binary search. Time complexity O(logn)
    if high < low:
        return -1
    mid = (low+high)//2
    if key == arr[mid]:
        return mid
    if key > arr[mid]:
        return find_key_index(arr, mid+1, high, key)
    return find_key_index(arr, low, mid-1, key)

def get_key_index(arr, arr_len, key):
    pivot = find_pivot_index(arr, 0, arr_len-1)
    print("Pivot: {:d}".format(pivot))
    if pivot == -1:  # array is not rotated and is in ascending order
        return find_key_index(arr, 0, arr_len-1, key)

    # if pivot is found, search for key slement in either of two sub-arrays
    if arr[pivot] == key:
        return pivot
    if arr[0] <= key:  # left size subarray
        return find_key_index(arr, 0, pivot-1, key)
    return find_key_index(arr, pivot+1, arr_len-1,  key)  # right  side subarray

def find_sqrt(n):
    # base case
    if n == 0 or n == 1:
        return 1

    low = 1
    high = n
    while (low <= high):
        mid = (low + high) // 2
        if mid * mid == n:
            return mid

        if (mid * mid < n):
            low = mid+1
            ans = mid
        else:
            high = mid-1

    return ans






# arr = [1, 3, 5, 7]
# len_arr = len(arr)
# k = 4
# print("Index of inserting element in sorted array", get_insert_position2(arr, len_arr, k))
#arr = [1, 3, 5, 5, 5, 5, 67, 123, 125]
#arr_len = len(arr)
#k = 125
#find_first_last(arr, arr_len, k)
#print('First occurrence of {:d} is {:d}'.format(k, find_first_occ(arr, arr_len, k, 0, arr_len - 1)))
#print('Last occurrence of {:d} is {:d}'.format(k, find_last_occ(arr, arr_len, k, 0, arr_len - 1)))
#arr  = [5, 6, 7, 8, 9, 10, 1, 2, 3]
#arr_len = len(arr)
#key = 1
#print("Index of key: {:d}".format(get_key_index(arr, arr_len, key)))
num = 11
print("Floor of square root is {:d}".format(find_sqrt(num)))
