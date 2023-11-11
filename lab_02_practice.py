# -*- coding: utf-8 -*-
"""lab 02 practice

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DIwfgmum7n6f2WuoYWAQZtNprW2p-aiA
"""

my_list = [1,2,3,4]

for i in my_list:
  i = 2 * i

print(my_list)

def sum_list(L):
  sum = 0
  for i in L:
    sum += i
  return sum


sum_list([1,3,7,-13])

def list_relu(L):
  new_list = [ ]
  for i in L:
    if i >= 0:
      new_list.append(i)
    elif i < 0:
      new_list.append(0)
  return new_list

list_relu([1, -2, 17, -3.2, -15])

import numpy as np  # Importing NumPy

my_var = (np.exp(5.0) - np.log(np.sqrt(5.0)))/(np.exp(np.cos(3.0)))

print(my_var)

v, u, w = np.array([1,3,-2,4,5]), np.array([1,1,-2,1,1]), np.array([1,0,1,0,1])

value = ((np.dot(v,u)/np.dot(u,u)) * u) + (((np.dot(v,w))/np.dot(w,w))*w)
print(value)

def first_rpt(M):
  new_matrix=M.copy()  # Remember to create a copy of your matrix.  After this lab you'll need to remember to do it on your own.
  for i in range(len(new_matrix)):
    if i > 0:
      new_matrix[i] = new_matrix[0]


  return new_matrix # Put your return value here.


first_rpt(np.array([[1,2,3,4], [-5,6,-7,-8], [1,5,2,3]]))

def matrix_sum(M):
  sum = 0
  for i in M: #i in M means i is each row in matrix M
    for j in i: #j in i means j is each value in row i
      sum += j #add that value to our sum placeholder
  return sum #return that sum value, which should have looped through every value in every row

matrix_sum(np.array([[1,-1,2,-3,1,1],[-2,-2,0,1,1,-5],[1,1,1,1,-2,-1]])) #ex., should return -5

def make_long_list(x):
  i, long_list = 1, []
  while i <= 100:
    long_list.append((x)**i)
    i += 1
  return long_list

make_long_list(.5)

def make_very_long_list(x):
  i, j, very_long_list = 1, 1, []
  while i < 100:
    j = 1
    while j <= 3:
      very_long_list.append((j)**i)
      j += 1
    i += 1
  return very_long_list


very_long_list = make_very_long_list(1)