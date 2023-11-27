LabID = "Lab11"
try:
    from graderHelp import ISGRADEPLOT
except ImportError:
    ISGRADEPLOT = True

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""**Downloading dataset**"""
df = pd.read_csv('Lab11data.csv', header=None)
X_neg = df.loc[df[100] == 0].drop(columns=100).values.transpose()
X_pos = df.loc[df[100] == 1].drop(columns=100).values.transpose()
X_total = df.loc[df[100] >= 0].drop(columns=100).values.transpose()
Alice = df.loc[df[100] < 0].drop(columns=100).values[0, :]
Bob = df.loc[df[100] < 0].drop(columns=100).values[1, :]

first_name="Benjamin"
last_name="Andreasen"
section_number="001"
BYUNetID="bca44"


# PROBLEM 1
def projection_coordinate(u, x):
    """
    :param u: vector basis for projection of x
    :param x: vector to be projected
    :return: coordinate of x projected to the line spanned by u

    >>> u_matrix = np.array([1/np.sqrt(6),1/np.sqrt(6),2/np.sqrt(6)])
    >>> x_matrix = np.array([2, 1, -3])
    >>> projection_coordinate(u_matrix,x_matrix)
    -1.2247448713915892
    """
    dot_product = np.dot(u, x)
    u_dot_u = np.dot(u, u)
    return dot_product / u_dot_u


# PROBLEM 2
def projection_2D(u1, u2, X):
    """
    :param u1: k-dimensional NumPy vector
    :param u2: k-dimensional NumPy vector
    :param X: k x p NumPy matrix
    :return: the coordinates of the points in X projected to the plane spanned by u1 and u2.

    >>> u1_matrix = np.array([1/3, 2/3, 2/3])
    >>> u2_matrix = np.array([0,-1/np.sqrt(2),1/np.sqrt(2)])
    >>> X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> projection_2D(u1_matrix,u2_matrix,X)
    array([[ 7.66666667,  9.33333333, 11.        ],
           [ 2.12132034,  2.12132034,  2.12132034]])
    """
    projections = np.zeros((2, X.shape[0]))

    for i, x in enumerate(X.T):
        proj_on_u1 = projection_coordinate(u1, x) * u1
        proj_on_u2 = projection_coordinate(u2, x) * u2

        projections[0, i] = np.dot(proj_on_u1, u1)
        projections[1, i] = np.dot(proj_on_u2, u2)

    return projections


# PROBLEM 3
# Compute the covariance matrix of our total dataset (X_total) using equation (4)
# Equation (4): W = (1/n-1) * X * X^T
n = X_total.shape[1]
W = (1 / (n - 1)) * np.dot(X_total, X_total.T)


# PROBLEM 4
# Compute eigenvalues and eigenvectors of the covariance matrix W

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(W)

# Sort the eigenvalues in descending order and get the indices
sorted_indices = np.argsort(eigenvalues)[::-1]

# Reorder the eigenvalues and eigenvectors based on sorted indices
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Save the three largest eigenvalues
L1, L2, L3 = sorted_eigenvalues[:3]

# Save the corresponding eigenvectors
u1, u2, u3 = sorted_eigenvectors[:,0], sorted_eigenvectors[:,1], sorted_eigenvectors[:,2]


# PROBLEM 5

total_variance = sum(eigenvalues)

reduced_variance = L1 + L2

relative_variance = reduced_variance / total_variance


# PROBLEM 6

# project X_neg and X_pos
X_neg_2D = projection_2D(X_neg, u1, u2)
X_pos_2D = projection_2D(X_pos, u1, u2)
# Project Alice and Bob
Alice_2D = projection_2D(Alice.reshape(-1, 1), u1, u2).flatten()
Bob_2D = projection_2D(Bob.reshape(-1, 1), u1, u2).flatten()


# PROBLEM 7
def plot_data(Z1=[], Z2=[], Z3=[], Z4=[]):
    """
    plot arrays of 2-dimensional data points
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    if len(Z1) > 0:
        Y1 = np.reshape(Z1, (2, -1))
        ax1.scatter(Y1[0, :], Y1[1, :], s=2, c='b', marker="o")
    if len(Z2) > 0:
        Y2 = np.reshape(Z2, (2, -1))
        ax1.scatter(Y2[0, :], Y2[1, :], s=2, c='r', marker="o")
    if len(Z3) > 0:
        Y3 = np.reshape(Z3, (2, -1))
        ax1.scatter(Y3[0, :], Y3[1, :], s=100, c='g', marker="o")
    if len(Z4) > 0:
        Y4 = np.reshape(Z4, (2, -1))
        ax1.scatter(Y4[0, :], Y4[1, :], s=100, c=[colors[7]], marker="o")
    plt.show()
    return None


# Save the values of your predictions below.
# +1 indicates the individual tests positive, while -1 indicates they test negative.
plot_data(X_neg_2D, X_pos_2D, Alice_2D, Bob_2D)

Alice_prediction = 0
Bob_prediction = 0


if __name__ == "__main__":
    pass
