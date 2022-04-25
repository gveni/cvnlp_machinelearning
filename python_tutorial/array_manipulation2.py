def get_prod_array_except_itself(arr):
    len_arr = len(arr)
    fwd_arr = [0]*len_arr
    bkwd_arr = [0]*len_arr

    fwd_arr[0] = 1
    bkwd_arr[len_arr-1] = 1

    # forward array calculation
    for i in range(1, len_arr):
        fwd_arr[i] = fwd_arr[i-1]*arr[i-1]

    # backward array calculation
    for i in range(len_arr-2, -1, -1):
        bkwd_arr[i] = bkwd_arr[i+1]*arr[i+1]

    # Take product of forward and backward array to get product of array except itself
    prod_arr = [0]*len_arr
    for i in range(len_arr):
        prod_arr[i] = fwd_arr[i] * bkwd_arr[i]

    print(prod_arr)

def find_histogram(arr):
    hist = {}
    count = 1
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]:
            count += 1
            hist[arr[i]] = count
        else:
            hist[arr[i]] = count
            count = 1
    if arr[len(arr)-1] != arr[len(arr)-2]:
        hist[arr[len(arr)-1]] = 1
    print(hist)

def find_freq_util(arr, low, high, freq):
    # if low-index element == high-index element, increment frequency of the element
    if arr[low] == arr[high]:
        freq[arr[low]] += high - low + 1
    else:
        # divide array into half and recurse
        mid = int(low+high)/2
        find_freq_util(arr, low, mid, freq)
        find_freq_util(arr, mid+1, high, freq)

def find_histogram2(arr):
    # initialize the frequency vector
    len_arr = len(arr)
    freq_arr = [0 for i in range(len_arr - 1 + 1)]
    find_freq_util(arr, 0, len_arr-1, freq_arr)

    # print result
    for i in range(0, (len_arr - 1)+1):
        if freq_arr[i] != 0:
            print("Frequency of", i, ":", freq_arr[i])

def first1sidx(arr, low, high):
    if high >= low:
        mid_idx = low + (high-low)//2  # get middle index
        # check if value at mid-index is first 1
        if (mid_idx == 0 or arr[mid_idx-1] == 0) and (arr[mid_idx] == 1):
            return mid_idx
        # if the value at mid-idx is 0, recur right-side
        elif arr[mid_idx] == 0:
            return first1sidx(arr, mid_idx+1, high)
        # if the value at mid-index is not first 1, recur left-side
        else:
            return first1sidx(arr, low, mid_idx-1)
    return -1

def get_row_max1s(mat):
    # Time complexity = O(mlogn)
    num_rows = len(mat)
    num_cols = len(mat[0])
    max = -1
    max_row_index = 0

    for i in range(num_rows):
        idx = first1sidx(mat[i], 0, num_cols-1)
        if (idx != -1) and (num_cols - idx) > max:
            max = num_cols - idx
            max_row_index = i
    return max_row_index


def get_row_max1s2(mat):
    num_r = len(mat)
    num_c = len(mat[0])
    max1_row_idx = 0
    idx = num_c - 1

    for i in range(0, num_r):
        flag = False
        while idx >= 0 and mat[i][idx] == 1:
            idx -= 1
            flag = True
            if flag:
                max1_row_idx = i
    return max1_row_idx

def get_longest_subseq_len(arr):
    arr = sorted(arr)
    print("Sorted array", arr)
    len_sub_arr1 = 1
    len_sub_arr2 = 1
    for i in range(1, len(arr)):
        if arr[i-1]+1 == arr[i]:
            len_sub_arr2 += 1
        else:
            len_sub_arr1 = len_sub_arr2
            len_sub_arr2 = 1
    print("Length of longest consecutive subsequence", len_sub_arr2)

def get_longest_subseq_len2(arr):
    # Hash array elements
    s = set()
    for ele in arr:
        s.add(ele)

    max_possible_len = 0
    # Check each possible sequence from start and update max_posssible_len
    for i in range(len(arr)):
        # if current array element is starting element of sequence
        if (arr[i] - 1) not in s:
            # check for next elements  in sequence
            j = arr[i]
            while (j in s):
                j += 1

            # Update max_possible_seq
            max_possible_len = max(max_possible_len, j - arr[i])

    print("Length of longest consecutive subsequence", max_possible_len)


# Driver function
#ip_array = [1, 1, 2, 2, 2, 3, 4, 10, 10, 10, 100]
#get_prod_array_except_itself(ip_array)
#find_histogram(ip_array)
#bool_mat = [[0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
#rowindex_max1s = get_row_max1s2(bool_mat)
#print("Row with maximum number of 1s", rowindex_max1s)
ip_arr = [45, 40, 46, 35, 44, 33, 34, 47, 43, 32, 42]
get_longest_subseq_len2(ip_arr)