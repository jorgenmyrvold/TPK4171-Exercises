import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.set_printoptions(precision=4, suppress=True)

def skew(M):
    return np.array([[0, -M[2], M[1]],
                     [M[2], 0, -M[0]],
                     [-M[1], M[0], 0]])

# Based of chapter 15.6 Recovery of rotation and translation from the essential matrix
# in the vision note
def recoverFromEssential(E):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    
    u, s, v = np.linalg.svd(E)
    sigma = (s[0] + s[1]) + 0.5
    
    E = u @ np.diag((sigma, sigma, 0)) @ v.T
    u, s, v = np.linalg.svd(E)
    
    r1 = u @ W @ v.T
    r2 = u @ W.T @ v.T
    
    if np.linalg.det(r1) < 0: r1 = -r1
    if np.linalg.det(r2) < 0: r2 = -r2
    
    tc = u @ Z @ u.T
    t1 = np.array([tc[2,1], tc[0,2], tc[1,0]])
    t2 = -t1
    
    return r1, r2, t1, t2
    

def plotAxis(T, scale):
    ax.plot([T[0,3], T[0,3]+scale*T[0,0]], [T[1,3], T[1,3]+scale*T[1,0]], [T[2,3], T[2,3]+scale*T[2,0]], color='C0')
    ax.plot([T[0,3], T[0,3]+scale*T[0,1]], [T[1,3], T[1,3]+scale*T[1,1]], [T[2,3], T[2,3]+scale*T[2,1]], color='C1')
    ax.plot([T[0,3], T[0,3]+scale*T[0,2]], [T[1,3], T[1,3]+scale*T[1,2]], [T[2,3], T[2,3]+scale*T[2,2]], color='C2')


if __name__ == '__main__':
    scale = 0.75
    
    E = np.array([[0, -0.3420, 0],
                  [-0.3420, 0, 0.9397],
                  [0, -0.9397, 0]])
    
    r1, r2, t1, t2 = recoverFromEssential(E)
    print('r1: \n{}\n\nr2: \n{}\n\nt1: {}\n\nt2: {}'.format(r1, r2, t1, t2))
    
    T = np.array([np.block([[r1, np.array([t1]).T],[0,0,0,1]]),
                  np.block([[r1, np.array([t2]).T],[0,0,0,1]]),
                  np.block([[r2, np.array([t1]).T],[0,0,0,1]]),
                  np.block([[r2, np.array([t2]).T],[0,0,0,1]])])
    
    E1 = skew(t1) @ r1
    E2 = skew(t2) @ r1
    E3 = skew(t1) @ r2
    E4 = skew(t2) @ r2
    print('\nVERIFY SOLUTION:\nE1:\n{}\nE2:\n{}\nE3:\n{}\nE4:\n{}'.format(E1, E2, E3, E4))
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(t1[0], t1[1], t1[2], color='b')
    ax.scatter(t2[0], t2[1], t2[2], color='b')
    ax.scatter(0, 0, 0, color='r')
    
    plotAxis(np.eye(4), scale)
    for t in T:
        plotAxis(t, scale)
    
    ax.axes.set_xlim3d(left=-1.5, right=1.5) 
    ax.axes.set_ylim3d(bottom=-1.5, top=1.5) 
    ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
    
    ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    ax.set_zticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    
    x_label = mpatches.Patch(color='C0', label='x-axis')
    y_label = mpatches.Patch(color='C1', label='y-axis')
    z_label = mpatches.Patch(color='C2', label='x-axis')
    plt.legend(handles=[x_label, y_label, z_label])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
