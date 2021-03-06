import pandas
import matplotlib.pyplot as plt
import seaborn
import numpy

def math(x, y):
    """ this function serves as a test run while i do this book.  This is a function that returns the sum of two variables: x and y"""

    return x + y



#The f function allows the user to more easily attach strings to one another
first_name = "Todd"
last_name = "Humphrey"

full_name = f"{first_name} {last_name}"

print(full_name)



#Couple ways to add to a list.  using .extend() is one way

my_list = [1, 2, 3, 4, 5, 6]

my_list.extend([7, 8, 9])

print(my_list)

#Or you could use addition

add_list = [10, 11]

my_list += add_list

print(my_list)

#Or, most frequently, you can append to a list one item at a time

my_list.append(12)

print(my_list)

#Common idiom in code that is used to pertain to a value you will throw away is the following

_, y1 = [1, 2]

print(y1)
#In this case, 1 is the value we do not care about, but we want to hold on to the y1 value
#You'll notice that y1 prints to "2", meaning it is saved as the value 2 from [1, 2]



#Tuples are just like lists, but you cannot modify a tuple

my_list1 = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4

print(my_list1)
print(my_tuple)

#Tuples are the way to go if you want to return multiple values from functions

def sum_and_product(x, y):
    return (x + y), (x * y)
#sum_and_product(2, 3) returns (5, 6)

s, p = sum_and_product(3, 4)
#s will equal 7, p will equal 12



#Dictionaries -- writing out a lot of the code provided by Joel Grus in an effort to help the dictionaries concept stick

grades = {"Joel": 80, "Tim": 95}
#In this scenario, the name is our 'key' and the grade is our value

joels_grade = grades["Joel"]
#In order to get a specific value, use the key in square brackets when referencing the dictionary

joel_has_grade = "Joel" in grades
#returns True, since "Joel" is in the dictionary "grades"

kates_grade = grades.get("Kate", 0)
#returns 0, since Kate does not have a grade in our dictionary.
#Second input in the function is the value to return if "get" cannot find the key in the dictionary

grades["Tim"] = 99 #Replaces tim's grade with a 99 instead of a 95
grades["Kate"] = 100 #adds a third entry for Kate

#defaultdict is a default dictionary which is useful for when you are trying to look up a key that the dictionary does not contain.
#It will add a value for it using a zero-argument function that you provide.
#Useful when using dictionaries to collect results by some key and not needing to check each time if a key already exists
from collections import defaultdict

#dd_list = defaultdict(list()) #list() produces an empty list
#dd_list[2].append(1) #Now dd_list contains {2: [1]}

dd_dict = defaultdict(dict) #dict() produces an empty dictionary
dd_dict["Joel"]["City"] = "Seattle" #{"Joel" : {"City": "Seattle"}}



#Sort is a good way to maintain clean lists
sort_list = [1, 2, 3, 4, 23, 5, 1 ,345, 23, 45, 5]
sort_list.sort()

print(sort_list)




#List comprehensions will be used a lot in the book.  Used to transform a list into anothe rlist by only choosing certain elements
even_numbers = [x for x in range(5) if x % 2 == 0] #[0, 2, 4]

#Can also be used to turn a list into a dictionary or a set
square_dict = {x: x * x for x in range(5)} #{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}



#Using tests: Important for making sure code doesn't fail, and when it does identifying where it is going wrong
#In the book, we will use "assert"

assert 1 + 1 == 2, "1 + 1 should equal 2, but it did not"
#Assert is the test, the second piece is the message to display if the test fails

#Though Joel Grus provides an explanation of classes in the crash course, I highly recommend going to the link below for more clarity when stuck
#https://docs.python.org/3/tutorial/classes.html



#Generators

def generate_range(n):
    i = 0
    while i < n:
        yield i #Every call to yield produces a value of the generator
        i += 1

for i in generate_range(10):
    print(f"i: {i}")

names = ["Todd", "John", "Ally", "Veronica"]

for i, name in enumerate(names):
    print(f"name {i} is {name}")
    #prints "name 0 is Todd, name 1 is John, and so on"
#Enumerate turns the values in a list into pairs (index, value)



#Randomness
#We will use random numbers often, but there are certain tricks to help us with it
import random
random.seed(10)
print(random.random())
random.seed(10)
print(random.random())
#By using .seed(), we will keep the same random numbers in order, so that we can continue to use those consistently
#random.random() produces numbers uniformly between 0 and 1
#for the usability for the random piece, look to page 35 in the book.  Very straightforward, and we are not trying to memorize functions

from typing import List
def total(xs: List[float]) -> float:
    return sum(total)

#Get used to Type Annotations, which allow you to more clearly define what type of data you are trying to input in your code

#Concludes the chapter!  Congrats Todd