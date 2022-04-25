from math import floor

# find missing number in array
def get_missing_num(arr):
    arr.sort()  # sort array first
    for n, _ in enumerate(arr):   # check if given element + 1 == next element
        if not arr[n]+1 == arr[n+1]:
            return arr[n]+1

def get_missing_num2(arr):
    # time complexity: O(n) No  auxilliary. Space complexity: O(1)
    len_arr = len(arr)
    tgt_sum = (len_arr+1)*(len_arr+2)//2
    actual_sum = sum(arr)
    return actual_sum - tgt_sum

def get_trail_zeros_factorial(num):
    # Get factorial
    factorial = 1
    for n in range(num):
        factorial *= n+1
    # Repeatedly divide the factorial by 10 till the remainder is 0 while incrementing its counter
    trailing_zeros = 0
    while factorial != 0:
        remainder = factorial % 10
        factorial //= 10
        if remainder != 0:
            return trailing_zeros
        else:
            trailing_zeros += 1

def get_trail_zeros_factorial2(num):
    # Prime factor logic
    #  trailing zeros depend upon combined occurrences of 2s and 5s
    # check for number of '5' occurrences
    # -ve number edge case
    if num < 0:
       return -1

    count = 0
    while (num >= 5):
        num //= 5
        count += num
    return count

ip_arr = [9, 2, 5, 6, 3, 7, 8]
# Find missing number in array
#missing_num = get_missing_num2(ip_arr)
#print("Missing number is", missing_num)
# Trailing zeros in factorial
num = 101
trail_zeros_factorial = get_trail_zeros_factorial2(num)
print("Number of training zeros of a factorial", trail_zeros_factorial)
