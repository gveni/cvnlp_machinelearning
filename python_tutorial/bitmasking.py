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

def get_num_set_bits(n):
    print(bin(n))
    num_set_bits = 0
    while n > 0:
        if n & 1 == 1:
            num_set_bits += 1
        n >>= 1
    print(f"Number of set bits in {n} are {num_set_bits}")


# Driver code
ip_num = 15
#print(get_binary_representation(ip_num))
#print(reverseBits(ip_num))
get_num_set_bits(ip_num)
