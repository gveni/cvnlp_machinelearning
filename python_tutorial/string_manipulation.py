from math import ceil

# Reverse a string
ip_str = 'Practicing python for interview'


def get_rvrs_str(given_str):
    rvrs_str = ''
    for i, _ in enumerate(given_str):
        rvrs_str += given_str[-i - 1]
    return rvrs_str


def get_rvrs_str2(given_str):
    rvrsd_str = ''
    for ch in given_str:
        rvrsd_str = ch + rvrsd_str
    return rvrsd_str


def get_rvrs_str3(given_str):
    return given_str[::-1]


def is_palindrome(given_str):
    # Run loop until half of string length
    for i in range(0, int(len(given_str) / 2)):
        if given_str[i] != given_str[len(given_str) - i - 1]:
            return False
    return True


def get_match_chars(ip_str1, ip_str2):
    matched_chars = []
    for i in ip_str1:
        if i in ip_str2:
            matched_chars.append(i)
    return len(matched_chars)


def get_v_and_c(ip_str):
    vowels = []
    consonants = []
    vowel_list = ['a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U']
    for i in ip_str:
        if i in vowel_list:
            if not i in vowels:
                vowels.append(i)
        else:
            if not i in consonants:
                consonants.append(i)
    return vowels, consonants


def remove_dup(ip_str):
    unique_char_str = ''
    for i, c in enumerate(ip_str):
        if not c in ip_str[i + 1:]:
            unique_char_str += c
    return unique_char_str


def twostrs_rotation_mutually(str1, str2):
    str1_1stchar = str1[0]
    if str1_1stchar in str2:
        str1_1stchar_pos = str2.find(str1_1stchar)
        new_str2 = str2[str1_1stchar_pos:] + str2[0:str1_1stchar_pos]
        if str1 == new_str2:
            return True
        else:
            return False
    else:
        return False


def reverse_word(s, start, end):
    while start < end:
        s[start], s[end] = s[end], s[start]
        start += 1
        end -= 1


def reverse_str_words(ip_str):
    ip_str = list(ip_str)  # convert str to list

    # reverse words in string separated by ' '
    start = 0
    while True:
        try:  # To avoid ValueError due to no ' ' after last character of string
            end = ip_str.index(' ', start)  # get index of first occurence  of ' '
            reverse_word(ip_str, start, end - 1)

            start = end + 1
        except ValueError:
            reverse_word(ip_str, start, len(ip_str) - 1)
            break

    # Reverse the entire string
    ip_str.reverse()

    # convert list back to string
    ip_str =  ''.join(ip_str)
    print(ip_str)


# Reverse a string
# rvrsd_str = get_rvrs_str3(ip_str)
# print("reversed string", rvrsd_str)
# Check if the given string is a palindrome
# print(f'{ip_str} is a palindrome: {is_palindrome(ip_str)}')
# Get number of matching characters between a pair of strings
ip_str1 = "abcdefg"
ip_str2 = "defgabc"
# print(f"Number of matched characters in {ip_str1} and {ip_str2} are {get_match_chars(ip_str1, ip_str2)}")
# Count number of vowels or consonants in a given string
# vowels, consonants = get_v_and_c(ip_str)
# vowels = remove_dups(vowels)  # remove duplicate elements from string
# print(f"Total of {len(vowels)} in '{ip_str}'. They are {vowels}")
# print(f"Total of {len(consonants)} in '{ip_str}'. They are {consonants}")
# Remove duplicate characters from string
# new_str = remove_dup(ip_str)
# print("Unique character string", new_str, 'of', ip_str)
# are_2strs_rotation_mutually = twostrs_rotation_mutually(ip_str1, ip_str2)
# print(f"Are two strings {ip_str1} and {ip_str2} mutually rotational", are_2strs_rotation_mutually)
ip_str = "I am a bad programmer"
reverse_str_words(ip_str)
