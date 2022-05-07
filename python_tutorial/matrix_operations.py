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
            rotated_mat[c][r] = ip_mat[r][(num_cols-1)-c]
    print("Rotated matrix anti clockwise:\n", rotated_mat)


def rotate_mat_clockwise(ip_mat):
    num_rows = len(ip_mat)
    num_cols = len(ip_mat[0])

    rotated_mat = [[0 for _ in range(num_rows)] for _ in range(num_cols)]
    for c in range(num_cols):
        for r in range(num_rows):
            rotated_mat[c][r] = ip_mat[num_rows-1-r][c]
    print("Rotated matrix clockwise\n", rotated_mat)


ip_mat = [[1, 2, 3, 10],
          [4, 5, 6, 11],
          [7, 8, 9, 12]]
# print("Input matrix", ip_mat)
# rotate_diagonals(ip_mat)
# bool_ip_mat = [[0, 0, 0, 0],
#               [0, 1, 0, 0],
#               [0, 1, 0, 0],
#               [0, 0, 1, 1]]
# print("Input boolean matrix", bool_ip_mat)
# print("Boolean mat:\n", get_boolean_mat(bool_ip_mat))
rotate_mat_anti_clockwise(ip_mat)
rotate_mat_clockwise(ip_mat)
