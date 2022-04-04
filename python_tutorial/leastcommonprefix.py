'''
Find the least common prefix from a list of strings
'''
str_list = ['aad_1', 'aadb_4', 'aad', 'aadbbc', 'aad2']

# Find the minimum length string
def get_len_smalleststr(str_list):
    len_smallest_str = 1000
    for s in str_list:
        if len(s) < len_smallest_str:
            len_smallest_str = len(s)
    return len_smallest_str

len_smallest_str = get_len_smalleststr(str_list)
print(f'Length of smallest string in the list', len_smallest_str)

def get_least_common_prfx(str_list):
    lcp = ''
    for l in range(len_smallest_str):  # for each character along the length of shortest string
        for s in str_list[1:]:  # for each string of list
            ref_char = str_list[0][l]
            if ref_char == s[l]:
                pass
            else:
                return lcp
        lcp += ref_char
    return lcp

least_common_prfx = get_least_common_prfx(str_list)
print("Least common prefix", least_common_prfx)