#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
np.set_printoptions(precision=4, suppress=True)


# # Exercise 3

# ## Problem 1

# In[26]:


def Skew(M):
    return np.array([[0, -M[2], M[1]],
                     [M[2], 0, -M[0]],
                     [-M[1], M[0], 0]])

x = np.array([[0, 0, 1],
              [0, 1, 1],
              [0, 2, 1],
              [0, 3, 1],
              [0.5, 0.5, 1],
              [-0.5, 0.5, 1],
              [-1, 1, 1],
              [-1, 2, 1],
              [-1, 3, 1]])

xp = np.array([[637, 509, 1],
                [639, 251, 1],
                [641, 108, 1],
                [638, 21, 1],
                [951, 361, 1],
                [331, 358, 1],
                [119, 252, 1],
                [240, 110, 1],
                [314, 25, 1]])

# Pick out 4 linearly independent points used to calculate the homography
xp_indep = np.block([[xp[0]],
                    [xp[1]],
                    [xp[4]],
                    [xp[5]]])

x_indep = np.block([[x[0]], [x[1]], [x[4]], [x[5]]])

def rc2H(x, xp):
    A1 = np.hstack([x[0,0]*Skew(xp[0]), x[0,1]*Skew(xp[0]), x[0,2]*Skew(xp[0])])
    A2 = np.hstack([x[1,0]*Skew(xp[1]), x[1,1]*Skew(xp[1]), x[1,2]*Skew(xp[1])])
    A3 = np.hstack([x[2,0]*Skew(xp[2]), x[2,1]*Skew(xp[2]), x[2,2]*Skew(xp[2])])
    A4 = np.hstack([x[3,0]*Skew(xp[3]), x[3,1]*Skew(xp[3]), x[3,2]*Skew(xp[3])])
    A = np.vstack((A1, A2, A3, A4))
    u,s,v = np.linalg.svd(A)
    h = v[8]/v[8,8]
    H = np.block([[h[0:3]], [h[3:6]], [h[6:9]]]).T 
    return H

H = rc2H(x_indep, xp_indep)
print(H)

# TEST
x_est = (H @ x.T).T
print("\n-------TEST-------")
for i in range(len(xp)):
    print(xp[i], '\t', x_est[i])


# Ser at resultatene ikke stemmer helt over ens, men de er i nærheten. I oppgave b under finner vi en bedre tilnærming av homografien H.

# ## Task 1b

# In[27]:


x = np.array([[0, 0, 1],
              [0, 1, 1],
              [0, 2, 1],
              [0, 3, 1],
              [0.5, 0.5, 1],
              [-0.5, 0.5, 1],
              [-1, 1, 1],
              [-1, 2, 1],
              [-1, 3, 1]])

xp = np.array([[637, 509, 1],
                [639, 251, 1],
                [641, 108, 1],
                [638, 21, 1],
                [951, 361, 1],
                [331, 358, 1],
                [119, 252, 1],
                [240, 110, 1],
                [314, 25, 1]])

xp_H = np.block([[xp[0]],
               [xp[1]],
               [xp[4]],
               [xp[5]]])

x_H = np.block([[x[0]], [x[1]], [x[4]], [x[5]]])

x = x.T
xp = xp.T
H = rc2H(x_H, xp_H)


def residual_transfer_error(hh):
    Hh = hh.reshape(3,3)
    xph = Hh @ x
    ph = xph[0:2]/xph[2,:]
    residual = p - ph
    return residual.flatten()


p = xp[0:2]

# Optimization of transfer error with DLT solution as initial value
# xp = np.block([[x_pix[0]],
#                [x_pix[1]],
#                [x_pix[4]],
#                [x_pix[5]]])

H0 = H
h0 = H0.flatten()
r0 = residual_transfer_error(h0)
d_DLT = r0 @ r0
res_1 = least_squares(residual_transfer_error, h0)

hopt = res_1.x
r_opt = residual_transfer_error(hopt) 
d_opt = np.dot(r_opt,r_opt)
Hopt = hopt.reshape(3,3)
Hopt = Hopt/Hopt[2,2]*Ha[2,2]
print('\n optimality = ', res_1.optimality)
print('\n Hopt = \n {} \n\n H = \n {}\n\n Ha = \n {}     \n\n d_DLT = {} \n d_opt = {}'
    .format(Hopt/2, H, Ha/2, d_DLT, d_opt))

