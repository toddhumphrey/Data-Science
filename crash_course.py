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
