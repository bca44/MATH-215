import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx

LabID="Lab10"

try:
  from graderHelp import ISGRADEPLOT
except ImportError:
  ISGRADEPLOT = True

"""**Enter your name, section number, and BYU NetID**"""

# Enter your first and last names in between the quotation marks.

first_name="Benjamin"

last_name="Andreasen"

# Enter your Math 215 section number in between the quotation marks.

section_number="Your Math 215 section number goes here"

# Enter your BYU NetID in between the quotation marks.  NOT YOUR BYU ID NUMBER!

BYUNetID="bca44"

df = pd.read_csv('Lab10webpagedata.csv')
webpagedata = df.values
webpage_data = np.array(webpagedata)
edges = [(i[0],i[1]) for i in webpagedata]
G = nx.from_edgelist(edges)
nx.draw(G,node_size=75)

E1=np.array([[0,1],[1,4],[0,2],[0,4],[1,3],[2,0],[2,4],[3,4],[3,2],[3,2]])
A2= np.array([[0,1,0,0,0,0,0,0],
 [0,0,1,0,0,0,0,0],
 [1,0,0,1,0,1,1,0],
 [0,0,0,0,1,1,0,0],
 [1,0,1,0,0,0,0,0],
 [1,0,0,0,0,0,0,0],
 [0,0,0,0,0,1,0,1],
 [1,0,0,0,0,0,0,0]])
P = np.array([[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [1/4, 0, 0, 1/4, 0, 1/4, 1/4, 0],
              [0, 0, 0, 0, 1/2, 1/2, 0, 0], [1/2, 0, 1/2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1/2, 0, 1/2], [1, 0, 0, 0, 0 ,0, 0, 0]])
E2 = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [2, 5], [2, 6], [3, 5], [3, 4], [4, 2], [4, 0], [5, 0], [6, 5], [6, 7],[7, 0]])


def adj_matrix(n,edge_list):
    """
    PROBLEM 1
    Write a function that takes in a number n and an edge list and returns the adjacency matrix of the graph.
    :param n: number of vertices
    :param edge_list: list of edges
    :return: adjacency matrix of the graph
    >>> adj_matrix(5, E1)
    array([[0, 1, 1, 0, 1],
          [0, 0, 0, 1, 1],
          [1, 0, 0, 0, 1],
          [0, 0, 2, 0, 1],
          [0, 0, 0, 0, 0]])
    """
    adj = np.zeros((n, n))

    for edge in edge_list:
        source, target = edge
        adj[source][target] += 1
    return adj.astype(int)


def degree_cent(n,edge_list):
    """
    Problem 2

    returns 1-dimensional NumPy array deg_array, whose i_th
    entry is the in-degree of the i_th vertex of the graph defined by the list edge_list

    :param n: number of vertices
    :param edge_list: list of edges
    :return: deg-array
    >>> degree_cent(5, E1)
    array([1, 1, 3, 1, 4])
    """
    deg_array = np.zeros(n, dtype=int)  # Initialize an array to store in-degrees

    for _, target in edge_list:
        deg_array[target] += 1  # Increment the in-degree of the target vertex

    return deg_array


# Problem 3
# Use the degree_cent function to calculate in-degrees
in_degrees = degree_cent(len(webpage_data), webpage_data)

# Use np.argsort to get the indices that would sort in_degrees in ascending order
sorted_indices = np.argsort(in_degrees)

# Reverse the sorted indices to get them in descending order
top_indices = np.flip(sorted_indices)

# The website with the highest in-degree is at the first index (top_indices[0])
top_indegree = top_indices[0]


def stoch_mat(A):
    """
    Problem 4
    returns stochastic matrix taken from matrix A

    :param A: square NumPy array
    :return: stochastic matrix
    >>> stoch_mat(A2)
    array([[0.  , 0.  , 0.25, 0.  , 0.5 , 1.  , 0.  , 1.  ],
       [1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 1.  , 0.  , 0.  , 0.5 , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.5 , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.25, 0.5 , 0.  , 0.  , 0.5 , 0.  ],
       [0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.  ]])
    """
    copy_matrix = np.transpose(A.copy())

    # Calculate the sum of entries in each column
    column_sums = np.sum(copy_matrix, axis=0)

    # Divide each column by the sum of its entries to make it stochastic
    stochastic_matrix = copy_matrix / column_sums

    return stochastic_matrix


def stoch_eig(P,k): # TODO: Fix this function
    """
    Problem 5

    :param P: stochastic matrix
    :param k: number of iterations
    :return: dominant eigenvector of P after k iterations

    >>> stoch_eig(P, 100)
    array([0.22727273, 0.22727273, 0.24242424, 0.06060606, 0.03030303,
    0.12121212, 0.06060606, 0.03030303])
    """
    n = P.shape[0]  # Get the size of the matrix (assuming it's square)

    # Initialize an initial guess for the eigenvector
    x = np.ones(n) / n

    # Perform power iteration for k iterations
    for _ in range(k):
        x = P @ x  # Multiply P by the current eigenvector estimate

    # Normalize the resulting eigenvector
    x /= np.linalg.norm(x)

    return x


def PageRank_cent(n,edge_list,k): # TODO: Fix this function
    """
    Problem 6

    Define a function called PageRank_cent(n,edge_list,k) which accepts as input a number of vertices n, a list of edges edge_list (which is formatted as a 2-dimensional NumPy array as in Problem 1), and an iteration number k, and performs the following:

Creates the adjacency matrix A of the network with n vertices and edges in edge_list,
Creates the stochastic matrix P corresponding to the adjacency matrix A (you may assume here that none of rows of A sum to zero),
Approximates the dominant eigenvector of P after k iterations.
Your function should return the approximation of the dominant eigenvector after k iterations (formatted as a 1-dimensional NumPy array).
    :param n: number of vertices
    :param edge_list:
    :param k:
    :return:

    >>> PageRank_cent(8, E2, 100)
    array([0.22727273, 0.22727273, 0.24242424, 0.06060606, 0.03030303,
    0.12121212, 0.06060606, 0.03030303])
    """
    A = adj_matrix(n, edge_list)

    # Create the stochastic matrix P
    row_sums = A.sum(axis=1)
    P = A / row_sums[:, np.newaxis]  # Normalize rows to make it stochastic

    # Approximate the dominant eigenvector of P after k iterations
    eigenvector = stoch_eig(P, k)

    return eigenvector


"""**Problem 7**""" # TODO: fix this answer

page_rank = PageRank_cent(len(webpage_data), webpage_data, 100)

top_PageRank = max(page_rank)

"""**STOP!  BEFORE YOU SUBMIT THIS LAB:**  Go to the "Runtime" menu at the top of this page, and select "Restart and run all".  If any of the cells produce error messages, you will either need to fix the error(s) or delete the code that is causing the error(s).  Then use "Restart and run all" again to see if there are any new errors.  Repeat this until no new error messages show up.

**You are not ready to submit until you are able to select "Restart and run all" without any new error messages showing up.  Your code will not be able to be graded if there are any error messages.**

To submit your lab for grading you must first download it to your compute as .py file. In the "File" menu select "Download .py". The resulting file can then be uploaded to http://www.math.byu.edu:30000 for grading.
"""