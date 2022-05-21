from collections import defaultdict


def rotate_diagonals(ip_mat):
    # Create a dictionary of lists to store diagonal elements under same
    # difference keys
    diag_dict = defaultdict(list)
    num_rows = len(ip_mat)
    num_cols = len(ip_mat[0])
    for row in range(num_rows):
        for col in range(num_cols):
            diag_dict[row - col].append(ip_mat[row][col])
    print(diag_dict)

    # Sort diagonal dictionary list in descending order
    for i in diag_dict.keys():
        diag_dict[i].sort(reverse=True)
    print(diag_dict)

    # put sorted dictionary lists in respective diagonals
    for row in range(num_rows):
        for col in range(num_cols):
            ip_mat[row][col] = diag_dict[row - col].pop(0)

    print("Sorted diagonal matrix:\n", ip_mat)


def get_boolean_mat(bool_ip_mat):
    num_rows = len(bool_ip_mat)
    num_cols = len(bool_ip_mat[0])
    bool_row = [0] * num_rows
    bool_col = [0] * num_cols
    for r in range(0, num_rows):
        for c in range(0, num_cols):
            if bool_ip_mat[r][c] == 1:
                bool_row[r] = 1
                bool_col[c] = 1
    print("bool_row values", bool_row)
    print("bool_col values", bool_col)

    for r in range(0, num_rows):
        for c in range(0, num_cols):
            if bool_row[r] == 1 or bool_col[c] == 1:
                bool_ip_mat[r][c] = 1
    return bool_ip_mat


def rotate_mat_anti_clockwise(ip_mat):
    num_rows = len(ip_mat)
    num_cols = len(ip_mat[0])
    rotated_mat = [[0 for _ in range(num_rows)] for _ in range(num_cols)]
    for c in range(num_cols):
        for r in range(num_rows):
            rotated_mat[c][r] = ip_mat[r][(num_cols - 1) - c]
    print("Rotated matrix anti clockwise:\n", rotated_mat)


def rotate_mat_clockwise(ip_mat):
    num_rows = len(ip_mat)
    num_cols = len(ip_mat[0])

    rotated_mat = [[0 for _ in range(num_rows)] for _ in range(num_cols)]
    for c in range(num_cols):
        for r in range(num_rows):
            rotated_mat[c][r] = ip_mat[num_rows - 1 - r][c]
    print("Rotated matrix clockwise\n", rotated_mat)


def get_row_max1s(bool_mat):
    # naive approach: Time complexity: O(m x n)
    num_rows = len(bool_mat)
    num_cols = len(bool_mat[0])
    row_max1s = -1
    col_starting1 = -1
    for r in range(num_rows):
        for c in range(num_cols):
            if bool_mat[r][c] == 1:
                if (num_cols - 1 - c) > col_starting1:
                    col_starting1 = num_cols - 1 - c
                    row_max1s = r
                    break
    print("Row with max 1s", row_max1s)


def get_first1_idx(arr, low, high):
    if high >= low:
        # Get middle  index
        mid = (low + high) // 2
        if (mid == 0 or arr[mid - 1] == 0) and arr[mid] == 1:  # Check if the element at middle index is  first 1
            return mid
        elif arr[mid] == 0:  # if element at middle  index  is 0, go to right side
            return get_first1_idx(arr, mid + 1, high)
        else:  # otherwise left side
            return get_first1_idx(arr, low, mid - 1)
    return -1


def get_row_max1s2(bool_mat):
    # Binary search: Time complexity(O(m x log(n)))
    num_rows = len(bool_mat)
    num_cols = len(bool_mat[0])
    row_max1s = -1
    col_starting1 = -1
    for r in range(num_rows):
        first1_idx = get_first1_idx(bool_mat[r], 0, num_cols - 1)
        print("first1 index:", first1_idx)
        if (num_cols - 1 - first1_idx) > col_starting1:
            col_starting1 = num_cols - 1 - first1_idx
            row_max1s = r
    print("Row with max 1s", row_max1s)


def find_element_idx(ip_mat, k):
    # Divide and conquer strategy
    # Time complexity: O(m) where n is #(rows)
    # Space complexity: O(1)
    num_rows = len(ip_mat)
    num_cols = len(ip_mat[0])
    i = 0
    j = num_cols - 1

    while i < num_rows and j >= 0:
        if ip_mat[i][j] == k:
            print("Desired element found at", i, j)
            return 1

        if ip_mat[i][j] > k:
            j -= 1
        else:
            i += 1

    print("Desired element not found")  # i < 0 or j > n-1
    return 0


ip_mat = [[1, 2, 3, 10],
          [4, 5, 6, 11],
          [7, 8, 9, 12],
          [14, 16, 17, 28],
          [15, 25, 30, 32]]
# print("Input matrix", ip_mat)
# rotate_diagonals(ip_mat)
# bool_ip_mat = [[0, 0, 1, 1],
#               [0, 1, 1, 1],
#               [0, 0, 1, 1],
#               [1, 1, 1, 1]]
# print("Input boolean matrix", bool_ip_mat)
# print("Boolean mat:\n", get_boolean_mat(bool_ip_mat))
# rotate_mat_anti_clockwise(ip_mat)
# rotate_mat_clockwise(ip_mat)
# get_row_max1s2(bool_ip_mat)
find_element_idx(ip_mat, k=19)
