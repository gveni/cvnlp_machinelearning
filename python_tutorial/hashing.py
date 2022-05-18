def get_two_sum_pair(ip_arr, two_sum):
    # Naive approach: Time complexity: O(n^2). Auxiliary space: O(1)
    two_sum_pair = []
    for i in range(len(ip_arr)-1):
        for j in range(i+1, len(ip_arr)):
            if ip_arr[i]+ip_arr[j] == two_sum:
                two_sum_pair = [ip_arr[i], ip_arr[j]]
                break
    if len(two_sum_pair) == 0:
        print(f"No pairs exist whose sum is {two_sum}")
    else:
        print(f"Pairs with sum {two_sum}:", two_sum_pair)

def get_two_sum_pair2(ip_arr, two_sum):
    # Hashmap. Time complexity: O(n). Space complexity: O(n)
    hash_map = {}
    for i in range(0, len(ip_arr)):
        second_num = two_sum - ip_arr[i]
        if second_num in hash_map:
            print(f"Pairs with sum {two_sum}:", ip_arr[i], second_num)
        hash_map[ip_arr[i]] = i


ip_arr = [0, -1, 2, -3, 1]
two_sum = -2
get_two_sum_pair2(ip_arr, two_sum)

