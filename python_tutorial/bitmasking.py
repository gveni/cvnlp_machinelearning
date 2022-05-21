# function to reverse
# bits of a number


def get_binary_representation(n):
    n = n << 4
    return n



def reverseBits(n):
    rev = 0
    # traversing bits of 'n' from the right
    while n > 0:
        # bitwise left shift 'rev' by 1
        rev = rev << 1
        # if current bit is '1'
        if n & 1 == 1:
            rev = rev ^ 1
        # bitwise right shift 'n' by 1
        n = n >> 1
    return rev


# Driver code
ip_num = 11
print(bin(ip_num))
print(get_binary_representation(ip_num))
#print(reverseBits(ip_num))
