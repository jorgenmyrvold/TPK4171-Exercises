import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

n = np.array([[0,0,1]])
pi = np.block([n, 0]).T
print(pi)
print(np.dot(pi, pi.T)) 

x = np.array([1,2,3])
print(np.cross(x,x))