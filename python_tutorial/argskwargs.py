def my_interests(*interests):
    print('My interests are: ' + ', '.join(interests))
    print('\n')

def my_favs(**favorites):
    print("My favorite things are:")
    for category, val in favorites.items():
        print(f'{category}: {val}')

def main():
    hobbies = ('bowling', 'hiking', 'running', 'learning_MLconcepts')
    # variable-length argument
    my_interests(*hobbies)  # enable to provide multiple hobbies as argument

    fav_things = {'Cuisine':'Indian', 'destination':'Hawaii', 'Sport':'Bowling', 'Fruit':'Pineapple'}
    # variable-length keyword argument
    my_favs(**fav_things)  # enable to provide multiple categories as argument

if __name__ == "__main__":
    main()

