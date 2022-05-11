def get_shortest_str_len(str_array):
    shortest_len = 1e5
    for s in str_array:
        if len(s) < shortest_len:
            shortest_len = len(s)
    return shortest_len

str_array = ['geeksforgeeks', 'geeks', 'geek', 'geezer']
shortest_str_len = get_shortest_str_len(str_array)
print(f"Length of shortest string is {shortest_str_len}")

ref_str = str_array[0][:shortest_str_len]
longest_common_prfx = ''
for i in range(shortest_str_len):
    flag = 1
    for s in str_array[1:]:
        if ref_str[i] != s[i]:
            flag = 0
            break
    if flag == 1:
        longest_common_prfx += ref_str[i]


print("Longest common prefix", longest_common_prfx)



