import numpy as np

LabID = "Lab9"

try:
    from graderHelp import ISGRADEPLOT
except ImportError:
    ISGRADEPLOT = True

"""**Enter your name, section number, and BYU NetID**"""

# Enter your first and last names in between the quotation marks.

first_name = "Benjamin"

last_name = "Andreasen"

# Enter your Math 215 section number in between the quotation marks.

section_number = "Your Math 215 section number goes here"

# Enter your BYU NetID in between the quotation marks.  NOT YOUR BYU ID NUMBER!

BYUNetID = "bca44"


def evect_approx1(x_0, k):
    """
    PROBLEM 1

    Define a function evect_approx1(x_0,k) which approximates the dominant eigenvector of A = [[1,1], [2, 0]]
    using the power method as described in section 9.1.

    :param x_0: 2-dimensional NumPy vector, representing an initial guess
    :param k: int
    :return: 2-dimensional NumPy vector x_k

    >>> evect_approx1(np.array([1, 9]), 10)
    array([3752, 3760])
    """
    A = np.array([[1, 1], [2, 0]])
    x_k = x_0

    for _ in range(k):
        # Apply the matrix A to the current vector x_k
        x_k = np.dot(A, x_k)

    return x_k


def eval_approx1(x_0, k):
    """
    PROBLEM 2

    Approximate the dominant eigenvalue of A = [[1,1], [2, 0]] using the power method
    For continuity of grading, compute your estimate of lambda_1 using the first entry of x_k+1 and x_k.

    :param x_0: initial guess 2-dimensional NumPy vector
    :param k: an integer
    :return: estimate of lambda_1

    >>> eval_approx1(np.array([1, 9]), 10)
    2.002132196162047
    """
    # Calculate the vector after k iterations
    x_k = evect_approx1(x_0, k)

    # Calculate the vector after k+1 iterations
    x_k_plus_1 = evect_approx1(x_0, k + 1)

    # Estimate lambda_1 using the first entry of x_k_plus_1 and x_k
    lambda_1_estimate = x_k_plus_1[0] / x_k[0] if x_k[0] != 0 else 0
    return lambda_1_estimate


def norm_evect_approx1(x_0, k):
    """
    PROBLEM 3

    Approximate the dominant eigenvector and eigenvalue of matrix A = [[1,1], [2, 0]],
    This time normalize the eigenvector at each step using the norm or length of the vector

    :param x_0: an initial guess
    :param k: an integer
    :return: the dominant eigenvector estimate x_k and an estimate of the eigenvalue

    >>> norm_evect_approx1(np.array([1, 9]), 10)
    (array([0.70635334, 0.70785942]), 1.995744680851064)
    """
    A = np.array([[1, 1], [2, 0]])
    x = x_0
    x_k_minus_1 = None  # Initialize the variable to store the second-to-last x

    for i in range(k):
        w = A @ x
        x = w / np.linalg.norm(w)

        # Store the vector from the second-to-last iteration
        if i == k - 2:
            x_k_minus_1 = x.copy()

    # Estimate eigenvalue using the first entry of w_k and x_{k-1}
    eigval_approx = w[0] / x_k_minus_1[0] if x_k_minus_1[0] != 0 else 0
    return x, eigval_approx


def norm_approx_gen(M, x_0, k):
    """
    PROBLEM 4

    Approximate the dominant eigenvector and  eigenvalue of any square matrix.

    Use the power iteration method described above to approximate
    the dominant eigenvector and eigenvalue of M up to the kth iterate.

    Normalize at each step by dividing the eigenvector approximation by the largest absolute value of its entries.

    Return the approximation of the eigenvector obtained in this way,
    as well as the eigenvalue approximation obtained by dividing the first entries of w_k and x_k-1.

    :param M: a square matrix of any size
    :param x_0: an initial guess
    :param k: an integer
    :return: the dominant eigenvector estimate x_k and an estimate of the eigenvalue

    >>> array1 = np.array([[2, 4, 6], [4, 8, 0], [1, 2, 9]])
    >>> array2 = np.array([1, 5, -1])
    >>> norm_approx_gen(array1, array2, 10) # TODO
    (array([0.98994349, 1. , 0.98491523]), 12.01744017467949)
    """
    x = x_0
    x_k_minus_1 = None  # To store the second-to-last x

    for i in range(k):
        w = M @ x
        x = w / np.linalg.norm(w, np.inf)  # Normalize by largest absolute value

        # Store the second-to-last vector
        if i == k - 2:
            x_k_minus_1 = x.copy()

    # Estimate the eigenvalue using the first entry of w and x_{k-1}
    eigval_approx = w[0] / x_k_minus_1[0] if x_k_minus_1[0] != 0 else 0
    return x, eigval_approx


def ray_quotient(M, x_0, k):
    """
    PROBLEM 5

    Applies power iteration, normalizing current eigenvector approximation at each step by the maximum absolute value.
    Then applies the Rayleigh quotient at the end to estimate the dominant eigenvalue of M.

    :param M: matrix
    :param x_0: initial guess
    :param k: integer
    :return: dominant eigenvalue approximation

    >>> array1 = np.array([[2, 4, 6], [4, 8, 0], [1, 2, 9]])
    >>> array2 = np.array([1, 5, -1])
    >>> ray_quotient(array1, array2, 10) # TODO maybe, only last two digits off
    12.001490204299047
    """
    x = x_0

    for i in range(k):
        w = M @ x
        x = w / np.linalg.norm(w, np.inf)  # Normalize by dividing by the largest absolute value of entries

    eigval_approx = (x @ M @ x) / (x @ x)

    return eigval_approx


"""
Problem 6

Compute x_3 and x_4 for the given matrix and initial vector
using the normalized power method function norm_approx_gen you defined in Problem 4.
Save the values of these vectors as x_vect_3 and x_vect_4 respectively.
"""

# Given matrix and initial vector
matrix = np.array([[3, 2, -2], [-1, 1, 4], [3, 2, -5]])
x_0 = np.array([1, 1, 1])

# Use norm_approx_gen to compute x_3 and x_4
x_vect_3, _ = norm_approx_gen(matrix, x_0, 3)
x_vect_4, _ = norm_approx_gen(matrix, x_0, 4)


def subscriber_vals(x_0, k):
    """
    PROBLEM 7

    Write a function subscriber_vals(x_0,k) that takes as input an initial state x_0 and a number of months k
    and returns the number of million subscribers for each service in the kth month using the model described above.

    Notice that the total number of subscribers each month should add up to the number of subscribers total in month 0.

    :param x_0: initial state
    :param k: number of months
    :return: number of millon subscribers for each service in the kth month

    >>> subscriber_vals(np.array([95, 102]), 10)
    array([ 78.81582031, 118.18417969])
    """
    # Define the transition matrix P based on the given probabilities
    P = np.array([[0.7, 0.2],  # Probabilities for Netflix
                  [0.3, 0.8]])  # Probabilities for Hulu

    # Compute P^k (the kth power of P)
    P_k = np.linalg.matrix_power(P, k)

    # Multiply x_0 by P_k to get the number of subscribers in the kth month
    x_k = np.dot(P_k, x_0)

    return x_k


""""
Problem 8

What is the proportion of Netflix subscribers to total customers after 6 months?
Save this quantity as the variable netflix_subs6 in your Colab notebook.
(This should be a value between 0 and 1.) You may use previously defined functions to solve this problem.
"""

# TODO - netflix_subs6 is not right

# Calculate the number of subscribers for both services after 6 months
subscribers_6_months = subscriber_vals(np.array([80, 120]), 6)

# Calculate the proportion of Netflix subscribers to total customers
netflix_subs6 = subscribers_6_months[0] / np.sum(subscribers_6_months)


"""
Problem 9

Construct the transition matrix describing the pond this bullfrog will be in, and save it as trans_matrix.
When ordering your variables to create the matrix, order the ponds as A, B, C, then D.
"""

trans_matrix = np.array([[0.80, 0.50, 0.30, 0.20],  # Probabilities for pond A
                         [0.05, 0.20, 0.10, 0.10],  # Probabilities for pond B
                         [0.10, 0.10, 0.30, 0.10],  # Probabilities for pond C
                         [0.05, 0.20, 0.30, 0.60]]) # Probabilities for pond D

