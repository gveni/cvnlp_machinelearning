
"""
Find minimum element in array
"""
def find_min_element(a):
    len_a = len(a)
    min_ele = a[0]
    for i in range(len_a):
        if min_ele > a[i]:
            min_ele = a[i]

    return min_ele


arr = [5, 7, 6, 10, 8, 3, 2, 4]
print("Minimum array element", find_min_element(arr) )