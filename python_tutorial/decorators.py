from math import sqrt
## Example 1
#def adder(x):
#    def adder2(y):
#        return x+y
#    return adder2
#
#adder_obj = adder(10)
#
#print("Adding two numbers based on first-class objects concept:", adder_obj(20))

## Example 2
# define a decorator
#def decorator_fn(func1, func2):
#    # inner_fn is a wrapper function inside which the argument 'func1' & 'func2' are called.
#    # This wrapper function can access all the outer functions as long as they are passed as arguments
#    print("Before getting into wrapper function")
#    def inner_fn(text):
#        print('I am in wrapper function')
#        func1(text)
#        func2(text)
#        print("After calling all outer functions")
#    return inner_fn
#
#def func_to_be_called1(text):
#    print(text.lower())
#
#def func_to_be_called2(text):
#    print(text.upper())
#
#sample_text = "Designing a decorator"
#print(sample_text)
#decorator_obj = decorator_fn(func_to_be_called1, func_to_be_called2)
#print("Initialized decorator object")
#decorator_obj(sample_text)

## Example 3
#def decorator_fn(func):
#    def adder(*args, **kwargs):
#        z = func(*args, **kwargs)
#        return z
#    return adder
#
#@decorator_fn
#def add_values(x, y):
#    return x + y
#
#x, y = 2, 3
#decorator_obj = decorator_fn(add_values)
#z = decorator_obj(x, y)
#print(x , "+", y, "=", z)

# Example 4: chaining decorators
def decor0(func):
    def inner1():
         x = func()
         return x * x
    return inner1

def decor1(func):
    def inner1():
        x = func()
        y = 4 * x
        return y
    return inner1

def decor2(func):
    def inner1():
        x = func()
        return sqrt(x)
    return inner1

@decor2
@decor1
@decor0
def num():
    return 10

print(num())  # calling decor2(decor1(decor0(num)))

