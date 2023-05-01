item_list = [
    ('product1', 10),
    ('product2', 8),
    ('product3', 12)
]

# Apply filter function through lambda function on every item of the iterable
# Task: filter list items whose prices >= 10
filtered_list = list(filter(lambda item: item[1] >= 10, item_list))
print(filtered_list)

# Apply map function through lambda  function on every item of the iterable
#  Create a new list with only prices
prices_list = list(map(lambda item: item[1]*item[1], item_list))
print(prices_list)