import numpy as np

timelist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

vallist1 = [-1, 2, 4, 6, 3, 0, -2, -4, -1, 0.5, 2]

vallist2 = [0, 0, 1, 2, 4, 6, 10, 15, 20, 15, 13]

A = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
B = np.array([[-1, 2, 4, 6, 3, 0, -2, -4, -1, 0.5, 2]])
C = np.array([[0, 0, 1, 2, 4, 6, 10, 15, 20, 15, 13]])

D = np.vstack([A, B])
E = np.vstack([D, C])

val = E[2]
val2 = val[5]

print(timelist[-3])