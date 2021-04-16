from typing import List

##Vectors

Vector = List[float]
height_weight_age = [70, #inches,
                    170, #pounds,
                    40 ] #years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4

#Vectors add componentwise, meanign if two vectors are the same length, their sum is the sum of their components
#v[0] + w[0], v[1] + w[1], and so on
#This functionality can be implemented using list comprehension to zip the vectors together

def add(v: Vector, w:Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "Vectors must have the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

#Same reasoning for subtracting two vectors

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtract corresponding elements"""
    assert len(v) == len(w), "Vectors must have the same length"
    return[v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

#Sometimes we will want to componentwise sum a list of vectors -- first element is the sum of all first elements, second the sum of all second, and so on

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

#We will need to multiply a vector by a scalar, which multiplies each element in a vector by the scalar

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

#Now we can compute the componentwise mean of a list of same-sized vectors

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

#A useful tool in linear algebra is the dot product
#If we have vectors v and w, the dot product is the length of the vector if we projected v onto w (see page 58)

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

#Now we can easily compute a vector's sum of squares

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1, v_2 * v_2, ... , v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

#Which can now be used to find its magnitude (length)

import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v)) #math.sqrt() is a square root function

assert magnitude([3, 4]) == 5

#We have what we need to find the distance between two vectors, which is defined as
#sqrt((v1 - w1)**2 + (v_2 - w_2) ** 2 + ... + (v_n - w_n) ** 2)

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance1(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))

def distance2(v: Vector, w: Vector) -> float:  # type: ignore
    return magnitude(subtract(v, w))

#Distance 1 and Distance 2 are equivalent

#For vectors in production, more likely better to use NumPy instead



##Matrices

#A matrix is a two-dimensional collection of numbers
#We will represent them as a list of lists, with each inner list having the same size and representing a row in the matrix
#If A is a matrix, A[i][j] is the element in the ith row and the jth column
#We use capital letters to denote a matrix

# Another type alias
Matrix = List[List[float]]

A = [[1, 2, 3],  # A has 2 rows and 3 columns
     [4, 5, 6]]

B = [[1, 2],     # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]

#Please note that in normal math, rows and columns would be 1-indexed.  But since we are using python, we will zero-index our rows and columns in a matrix
#Given the list-of-lists representation, a matrix's shape is the number of rows and number of columns
#len(A) rows and len(A[0]) columns

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_columns = len(A[0]) if A else 0 #number of elements in first row
    return num_rows, num_columns

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)

#a matrix with n rows and k columns will be referred to as an n x k matrix
#each row of an n x k matrix has a length of k
#each column of an n x k matrix has a length of n

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]             # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j]          # jth element of row A_i
            for A_i in A]   # for each row A_i

#We will end up wanting to create a matrix given its shape and a function for generating its elements using list comprehension

from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             # given i, create a list
             for j in range(num_cols)]  #   [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # create one list for each i


def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            user 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9

assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are not friends"

# only need to look at one row
friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]
