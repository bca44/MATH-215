# -*- coding: utf-8 -*-
"""lab 04 practice

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VU-qIC8zN2KOTWqsLZqL9p-72JK1-gA3

#**JACOBI'S METHOD**

Jacobi's method - estimating solutions for a system of linear equations, using iteration. You start with an intial estimate and plug it into the system. This gives you another estimate which, hopefully, will be closer to the actual solution.

PROBLEM 1: Solve system (1) above, by hand, and save the x value in a variable called x_val and the y value in a variable called y_val.
"""

# Replace the values of 0 with the values you solved for in Problem 1.

x_val=1

y_val=1

"""PROBLEM 2: Define a function, called jacobi1_iteration(x,y), which accepts as input values x and y (which we think of as being
 and
 respectively), and returns a list [new_x,new_y], where new_x is the updated value
 and new_y is the updated value
 using Jacobi’s method and equations (2) and (3).

In other words, jacobi1_iteration(x,y) should return the results of performing one iteration of Jacobi’s method for system (1) on the inputs x, y.
"""

# Performs one iteration of the Jacobi method for system (1) applied to the point (x,y).

def jacobi1_iteration(x,y):
  new_x = (1/7) * (6 + y)
  new_y = (1/5) * (4 + x)
  return [new_x, new_y]


jacobi1_iteration(3,5)

"""PROBLEM 3: Define a function, called jacobi1_method(n) which accepts as input a single non-negative integer n, and returns a list [x_n,y_n], where x_n and y_n are the values of Xn and Yn  respectively for Jacobi’s method when applied to system (1). Use (X0, Y0) = (0,0) as your starting approximation."""

# Performs n iterations of the Jacobi method on system (1) with starting estimate (0,0).

def jacobi1_method(n):
  x_n, y_n = 0, 0
  for i in range(n):
    [x_n, y_n] = jacobi1_iteration(x_n, y_n)
  return [x_n, y_n]


jacobi1_method(3)

"""PROBLEM 4:Use the function jacobi1_method (from problem 3) to answer the questions below.


1.   What is the smallest value of n so that Xn
 and Yn are both within 0.1 of the values for x_val and y_val respectively? Save the value of the smallest such n as a variable called n_var1.
2.   What is the smallest value of n so that Xn and Yn are both within 0.0001 of the values for x_val and y_val respectively? Save the value of the smallest such n as a variable called n_var2.


"""

# Replace the values of 0 with the values you solved for in Problem 4.


def find_n(error, max_n):
  # finds the minimum number of iterations, n, needed to reach the specified 'error' level or smaller
  # max_n specifies the max number of iterations that should be tested
  x_val, y_val = 1, 1
  for i in range(max_n):
    [x_n, y_n] = jacobi1_method(i)
    if (abs(x_n - x_val) <= error) and (abs(y_n -y_val) <= error):
      return i

n_var1= find_n(0.1, 50)

n_var2=find_n(0.0001, 50)

"""#**Gauss-Seidel Method**

We can change the above procedure to obtain a different method for solving systems of linear equations, called the Gauss-Seidel method. The only difference between these two methods is the timing of when we plug in the values of x_n-1
 and y_n-1
 to get x_n
 and y_n
 . In the Gauss-Seidel method we compute x_n
 by plugging y_n-1
 into equation (2),

(2) x = 1/7 (6 + y)

the same as we do in Jacobi's method. However, when finding y_n, we plug the newly computed value of x_n into equation (3),

(3) y = 1/5 (x + 4)

instead of plugging in x_n-1 as we would do in Jacobi's method. In other words, we always plug in the most recently computed values for x and y when computing x_n and y_n.
In the above example, starting with (x_0, y_0) = (0, 0) as before, we would compute
x_1 = 1/7 (6 + y_0) = 1/7 (6 + 0) = 6/7
and
y_2 = 1/5 (x_1 + 4) = 1/5 (6/7 + 4) = 34/35

PROBLEM 5: Define a function, called gs1_iteration(x, y), which accepts as input values for x and y, which we think of as being x_n-1 and y_n-1 respectively, and returns a list [new_x, new_y], where new_x is the udpated value x_n and new_y is the updated value y_n using the Gauss-Seidel method and equations (2) and (3).

In other words, gs1_iteration(x, y) should return the results of performing one iteration of the Gauss Seidel method for system (1) on the inputs x, y.
"""

def f(y):
  x = (1/7) * (6 + y)
  return x


def g(x):
  y = (1/5) * (x + 4)
  return y


def gs1_iteration(x,y):
# Performs one iteration of the Gauss-Seidel method for system (1) applied to the point (x,y).
  new_x = (1/7) * (6 + y)
  new_y = (1/5) * (4 + new_x)
  return [new_x, new_y]


gs1_iteration(3,5)

"""PROBLEM 6: Define a function, called gs1_method(n), which accepts as input a single non-negative integer n, and returns a list [x_n, y_n], where x_n and y_n are the values of x_n and y_n, respectively, for the Gauss Seidel method when applited to system (1) above. Use (x_0, y_0) = (0, 0) as your starting approximation."""

def gs1_method(n):
    # Performs n iterations of the Gauss-Seidel method on system (1) with starting estimate (0,0).
  x_n, y_n = 0, 0
  for i in range(n):
    [x_n, y_n] = gs1_iteration(x_n, y_n)
  return [x_n, y_n]


gs1_method(1)

"""PROBLEM 7: Use the function gs1_method(n) to answer the questions below.


1. What is the smallest value of n so that x_n and y_n are both within 0.1 of the values for x_val and y_val, respectively? Save the value of the smallest such n as a variable called n_var3.
2. What is the smallest value of n so that x_n and y_n are both within 0.0001 of the values for x_val and y_val, respectively? Save the value of the smallest such n as a variable called n_var4.


"""

# Replace the values of 0 with the values you solved for in Problem 4.


def find_n_gs1(error, max_n):
  # finds the minimum number of iterations, n, needed to reach the specified 'error' level or smaller, using the GS method and gs1_method() function
  # max_n specifies the max number of iterations that should be tested
  x_val, y_val = 1, 1
  for i in range(max_n):
    [x_n, y_n] = gs1_method(i)
    if (abs(x_n - x_val) <= error) and (abs(y_n -y_val) <= error):
      return i

n_var3= find_n_gs1(0.1, 50)

n_var4=find_n_gs1(0.0001, 50)

"""#**ERROR**

For every n, we can think of the values (x_n, y_n), which come either from Jacobi's method or the Gauss-Seidel method, as being an approximation to the solution of system (1). We can therefore measure how close our approximation is by computing the distance from the point (x_n, y_n) to the point (x_val, y_val).

If a and b are two NumPy arrays of the same size, then we can compute the distance between a and b by the command:
```
np.linalg.norm(a-b) # a and b must be NumPy arrays here, and not lists
```
Remember that you will need to import NumPy in this notebook before you can use the function **np.linalg.norm** or any of the NumPy functions in this lab. Another you may need to remember is how to convert lists to arrays. For example, if you have two lists **list1** and **list2** and you want to find the distance between them, *thinking of them as vectors*, you will need to cast them as NumPy arrays when plugging them into the **np.linalg.norm** function as follows:
```
np.linalg.norm(np.array(list1) - np.array(list2)))
```
As you probably guessed, when we plug the list **list1** into the funciton **np.array**, it returns a NumPy array with the same values as **list1**. In other words, **np.array** takes lists and converts them into arrays.

**PROBLEM 8**

Define a function **gs1_error(n)** that acceps as input a non-negative integer n and outputs the error of the Gauss-Seidel approximation (x_n, y_n) to the solution of system (1). In other words, it should output the distance between the *nth* approximation (x_n, y_n) and the true value (x_val, y_val) of the solution to (1).
"""

import numpy as np


def gs1_method(n):
    # Performs n iterations of the Gauss-Seidel method on system (1) with starting estimate (0,0).
  x_n, y_n = 0, 0
  for i in range(n):
    [x_n, y_n] = gs1_iteration(x_n, y_n)
  return [x_n, y_n]


def gs1_error(n):
    # Finds the error of the nth approximation of the solution to system (1) using the Gauss-Seidel method.
    (x_n, y_n) = gs1_method(n)
    return np.linalg.norm(np.array([x_n, y_n]) - np.array([x_val, y_val]))


gs1_error(3)

"""Create a graph the plots the error of the estimate (x_n, y_n) for the first 50 values of *n* (i.e., for n = 0, 1, ..., 49). We will provide you with the code you need to create the plot, but you should read through it to make sure you understand what each step is doing. You should be familiar with most of the coommands."""

# The following code will construct your plot of gs1_error for you.  You don't need to change anything in this cell, simply execute it. Consider this one a freebie.

# Note that you must have a function defined called gs1_error from the previous problem in order for the plot to be created.  We first import matplotlib.pyplot:

import matplotlib.pyplot as plt



# This command uses the function gs1_error to create a new function vect_gs1_error which will accept NumPy arrays of various sizes as input, instead of just a single number.

vect_gs1_error=np.vectorize(gs1_error)



# This creates a NumPy array of values of the form [0,1,2,...,48,49], similar to the np.linspace command.  The 1 in the function tells NumPy to count up by ones.

n_vals=np.arange(0,50,1)



# This creates the plot, and labels the axes.  See if you can determine what each command is doing.

plt.title('Error of the Gauss-Seidel Method Applied to System 1')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.plot(n_vals,vect_gs1_error(n_vals),'ro')
plt.show()

"""#**CONVERGANCE**

Lots of words... See canvas page lol

**PROBLEM 9**

Define functions **gs2_iteration(x, y)** and **gs2_method(n)**, similar to previous problems, this time to produce the estimates **(x_n, y_n)** as computed by the Gauss-Seidel method applied to system (4). Use **(x_0, y_0) = (0, 0)** again as the starting estimate.

Define an error function **gs2_error(n)** similar to previous problems, and create a plot that shows the error of the Gauss-Seidel estimates for the first 40 values of *n*.
"""

# Gives one iteration of the Gauss-Seidel method for system (4) applied to the point (x,y).

### system (4):
# x = y + 1
# y = 5 - 2x


def gs2_iteration(x,y):
# Performs one iteration of the Gauss-Seidel method for system (4) applied to the point (x,y).
  new_x = (1 + y)
  new_y = (5 - (2 * new_x))
  return [new_x, new_y]




# Performs n iterations of the Gauss-Seidel method on system (4) with starting estimate (0,0).

def gs2_method(n):
    # Performs n iterations of the Gauss-Seidel method on system (4) with starting estimate (0,0).
  x_n, y_n = 0, 0
  for i in range(n):
    [x_n, y_n] = gs2_iteration(x_n, y_n)
  return [x_n, y_n]






# Finds the error of the nth approximation of the solution to system (4) using the Gauss-Seidel method.

def gs2_error(n):
  # Finds the error of the nth approximation of the solution to system (4) using the Gauss-Seidel method.
  (x_n, y_n) = gs2_method(n)
  return np.linalg.norm(np.array([x_n, y_n]) - np.array([2, 1]))

gs2_method(5), gs2_error(3)

# The following code will construct your plot of gs2_error for you.  You don't need to change anything in this cell, simply execute it. Consider this one another freebie.

# Note again that you must have a function defined called gs2_error from the previous problem in order for the plot to be created.

vect_gs2_error=np.vectorize(gs2_error)

n_vals=np.arange(0,50,1)

plt.title('Error of the Gauss-Seidel Method Applied to System 4')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.plot(n_vals,vect_gs2_error(n_vals),'ro')
plt.show()

"""**PROBLEM 10**

Similar to the problems above, define functions called **gs3_iteration(x, y, z)** and **gs3_method(n)** which use the Gauss-Seidel method and starting approximation **(x_0, y_0, z_0) = (0, 0, 0)**
to solve the following system

*   **5x-2y+3z = -8**
*   **x + 4y + -4z = 102**
*   **-2x - 2y + 4z = -90**
"""

def gs3_iteration(x,y,z):
    # Performs one iteration of the Gauss-Seidel method for system above applied to the point (x, y, z).
  new_x = (1/5) * (-8 + (2 * y) - (3 * z))
  new_y = (1/4) * (102 - new_x + (4 * z))
  new_z = (1/4) * (-90 + (2 * new_x) + (2 * new_y))
  return [new_x, new_y, new_z]




# Performs n iterations of the Gauss-Seidel method on the final system with starting estimate (0,0,0).

def gs3_method(n):
  x_n, y_n, z_n = 0, 0, 0
  for i in range(n):
    [x_n, y_n, z_n] = gs3_iteration(x_n, y_n, z_n)
  return [x_n, y_n, z_n]


gs3_method(4)

### TODO fix this
# returns [-24.28107947500001, 68.84884399375002, -88.90488511751659], needs to be [10.74020625, 11.6154796875, -11.32215703125]

"""Does the GS method applied to this system seem to converge or diverge? Do your observations contradict **Theorem 1**?"""