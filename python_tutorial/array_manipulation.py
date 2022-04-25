from math import ceil


def rotate_array(arr, pos=3):
    return arr[pos:] + arr[:pos]


def rotate_array2(arr, pos=3):
    #  Time complexity: O(n*pos)
    # Auxiliary space: O(1)
    len_arr = len(arr)
    for p in range(pos):
        tmp = arr[0]
        for l in range(len_arr - 1):
            arr[l] = arr[l + 1]
        arr[len_arr - 1] = tmp
    return arr


def gcd(n, d):
    if d == 0:
        return n
    else:
        return gcd(d, n % d)


def rotate_array3(arr, pos=3):
    len_arr = len(arr)
    d = pos % len_arr
    g_c_d = gcd(len_arr, pos)
    print(f"GCD of {len_arr} and {pos} is {g_c_d}")
    for i in range(g_c_d):
        temp = arr[i]
        j = i
        while 1:
            k = j + d
            if k >= len_arr:
                k = k - len_arr
            if k == i:
                break
            arr[j] = arr[k]
            j = k
        arr[j] = temp

    return arr


def get_majoity_element(arr):
    arr.sort()
    if arr[ceil(len(arr) / 2 - 1)] == arr[-1]:
        return arr[-1]
    else:
        return -1


def get_plus_one(arr):
    l = len(arr)
    # Add 1 to last digit and check  for carry
    arr[l - 1] += 1
    carry = arr[l - 1] / 10
    arr[l - 1] = arr[l - 1] % 10
    # Traverse from second last digit
    for i in range(l - 2, -1, -1):
        if carry == 1:
            arr[i] += 1
            carry = arr[i] / 10
            arr[i] = arr[i] % 10
    if carry == 1:
        arr.insert(0, 1)


def get_alternate_polarities(a):
    len_a = len(a)
    for i in range(len_a - 1):
        j = i + 1  # for swapping
        k = j  # for iteration
        while 1:
            sum_a2 = a[i] + a[k]
            if sum_a2 > a[i] and sum_a2 > a[k]:
                k += 1
                if k == len_a:
                    break
            else:
                tmp = a[j]
                a[j] = a[k]
                a[k] = tmp
                break
    print("Input array with oppoite polarities", a)

def get_alternate_polarities2(a):
    a_minus, a_plus = [], []
    for i in range(len(a)):
        if a[i] < 0:
            a_minus.append(a[i])
        else:
            a_plus.append(a[i])

    b = []
    for i in range(max(len(a_minus), len(a_plus))):
        if i < len(a_minus):
            b.append(a_minus[i])
        if i < len(a_plus):
            b.append(a_plus[i])
    print("Input array with opposite polarities", b)



# ip_arr = [1, 2, 3, 4, 5, 6]
# num_ele2rotate = 4
# rotated_array = rotate_array3(ip_arr, num_ele2rotate)
# print("rotated array:", rotated_array)
# A majority element in an array is the one if it occurs > half number of times
# ip_arr = [3, 3, 4, 2, 4, 2, 4, 4]
# major_ele = get_majoity_element(ip_arr)
# if major_ele != -1:
#    print(f"Majority element {major_ele}")
# else:
#    print("No majority element")
# Add 1 to number represented as array of digits
# ip_arr = [9, 0, 0]
# get_plus_one(ip_arr)
# for i in range(len(ip_arr)):
#    print(ip_arr[i])
a = [4, 3, -2, 1, 2,  3, -7]
get_alternate_polarities2(a)
