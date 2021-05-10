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

def LS_Triangulate(x, xp, R, t):
    L1 = x
    L1p = np.array([0, 0, 0])
    L2 = R.T @ xp
    L2p = -np.cross(R.T@t, L2)
    
    # Calculation of closest point to the four lines
    A = np.block([[skew(L1)], [skew(L2)]])
    b = -np.block([L1p, L2p])
    
    ATA_inv = np.linalg.inv(A.T @ A)
    r = ATA_inv @ A.T @ b
    # r = np.linalg.inv(A.T @ A) @ A.T @ b
    rp = R@r + t
    return r, rp

def midpointTriangulate(x, xp, R, t):
    L1 = x
    L1p = np.array([0, 0, 0])
    L2 = R.T @ xp
    L2p = -np.cross(R.T@t, L2)
    
    # Common normal
    Ln = np.cross(L1, L2)
    Lnp = np.cross(L1, L2p) + np.cross(L1p, L2)
    u = np.cross(L1, Ln); u4 = np.dot(L1, Lnp) # Plane with line 1 and common normal
    v = np.cross(L2, Ln); v4 = np.dot(L2, Lnp) # Plane with line 2 and common normal
    
    # Intersection of lines and common normal
    x = -v4*L1 + np.cross(v, L1p); x4 = np.dot(v, L1)
    y = -u4*L2 + np.cross(u, L2p); y4 = np.dot(u, L2)
    # Midpoint
    r = (y4*x+x4*y)/(2*x4*y4)
    rp = R @ r + t
    return r, rp

def calculate_point(s1, s2, R1, R2, t1, t2):
    # r11,r21 = midpointTriangulate(s1,s2,R1,t1)
    # r12,r22 = midpointTriangulate(s1,s2,R1,t2)
    # r13,r23 = midpointTriangulate(s1,s2,R2,t1)
    # r14,r24 = midpointTriangulate(s1,s2,R2,t2)
    
    r11,r21 = LS_Triangulate(s1,s2,R1,t1)
    r12,r22 = LS_Triangulate(s1,s2,R1,t2)
    r13,r23 = LS_Triangulate(s1,s2,R2,t1)
    r14,r24 = LS_Triangulate(s1,s2,R2,t2)
        
    print('\n\nr11, r21 = {}\t| {}'.format(r11, r21))
    print    ('r12, r22 = {}\t| {}'.format(r12, r22))
    print    ('r13, r23 = {}\t| {}'.format(r13, r23))
    print    ('r14, r24 = {}\t| {}'.format(r14, r24))
    
    if   r11[2] > 0 and r21[2] > 0: return r11, R1, t1
    elif r12[2] > 0 and r22[2] > 0: return r12, R1, t2
    elif r13[2] > 0 and r23[2] > 0: return r13, R2, t1
    elif r14[2] > 0 and r24[2] > 0: return r14, R2, t2

def plotAxis(T, scale, ax, one_color=''):
    if one_color:
        ax.plot([T[0,3], T[0,3]+scale*T[0,0]], [T[1,3], T[1,3]+scale*T[1,0]], [T[2,3], T[2,3]+scale*T[2,0]], color=one_color)
        ax.plot([T[0,3], T[0,3]+scale*T[0,1]], [T[1,3], T[1,3]+scale*T[1,1]], [T[2,3], T[2,3]+scale*T[2,1]], color=one_color)
        ax.plot([T[0,3], T[0,3]+scale*T[0,2]], [T[1,3], T[1,3]+scale*T[1,2]], [T[2,3], T[2,3]+scale*T[2,2]], color=one_color)
    else:
        ax.plot([T[0,3], T[0,3]+scale*T[0,0]], [T[1,3], T[1,3]+scale*T[1,0]], [T[2,3], T[2,3]+scale*T[2,0]], color='xkcd:bright purple')
        ax.plot([T[0,3], T[0,3]+scale*T[0,1]], [T[1,3], T[1,3]+scale*T[1,1]], [T[2,3], T[2,3]+scale*T[2,1]], color='xkcd:bright green')
        ax.plot([T[0,3], T[0,3]+scale*T[0,2]], [T[1,3], T[1,3]+scale*T[1,2]], [T[2,3], T[2,3]+scale*T[2,2]], color='xkcd:golden yellow')


def plot(T, points, one_color='', x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z_min=-1.5, z_max=1.5, x_intervall=0.5, y_intervall=0.5, z_intervall=0.5): # 
    '''
    T: list of all transformation to plot
    points: list of all points to plot
    '''
    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
    patches = []
    color_cycle = 0
    
    for i, t in enumerate(T):
        plotAxis(t, scale, ax, one_color=one_color)
        ax.scatter(t[0,3], t[1,3], t[2,3], color='C{}'.format(i))
        patches.append(mpatches.Patch(color='C{}'.format(color_cycle), label='t{}'.format(i+1)))
        color_cycle += 1
    
    for p in points:
        ax.scatter(p[0], p[1], p[2], color='C{}'.format(color_cycle))
        patches.append(mpatches.Patch(color='C{}'.format(color_cycle), label='p{}'.format(i+1)))
        color_cycle += 1
    
    ax.axes.set_xlim3d(left=x_min, right=x_max) 
    ax.axes.set_ylim3d(bottom=y_min, top=y_max) 
    ax.axes.set_zlim3d(bottom=z_min, top=z_max)
    
    ax.set_xticks(np.arange(x_min, x_max, x_intervall))
    ax.set_yticks(np.arange(y_min, y_max, y_intervall))
    ax.set_zticks(np.arange(z_min, z_max, z_intervall))
    
    if not one_color:
        patches.append(mpatches.Patch(color='xkcd:bright purple', label='x-axis'))
        patches.append(mpatches.Patch(color='xkcd:bright green', label='y-axis'))
        patches.append(mpatches.Patch(color='xkcd:golden yellow', label='z-axis'))
    plt.legend(handles=patches)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

if __name__ == '__main__':
    scale = 0.75
    
    E = np.array([[0, -0.3420, 0],
                  [-0.3420, 0, 0.9397],
                  [0, -0.9397, 0]])

    ### Recover R and t from E    
    r1, r2, t1, t2 = recoverFromEssential(E)
    print('r1: \n{}\n\nr2: \n{}\n\nt1: {}\nt2: {}'.format(r1, r2, t1, t2))
    print('|t1| = {}\n|t2| = {}'.format(np.linalg.norm(t1), np.linalg.norm(t2)))
    
    T = np.array([np.block([[r1, np.array([t1]).T],[0,0,0,1]]),
                  np.block([[r1, np.array([t2]).T],[0,0,0,1]]),
                  np.block([[r2, np.array([t1]).T],[0,0,0,1]]),
                  np.block([[r2, np.array([t2]).T],[0,0,0,1]])])
    
    E1 = skew(t1) @ r1
    E2 = skew(t2) @ r1
    E3 = skew(t1) @ r2
    E4 = skew(t2) @ r2
    print('\nVERIFY SOLUTION:\nE1:\n{}\nE2:\n{}\nE3:\n{}\nE4:\n{}'.format(E1, E2, E3, E4))
    
    for t in T:
        plot([np.eye(4), t], [], one_color='')

    ### Calculate point, and determine correct orientation
    s1 = np.array([0.0098, 0.1004, 1])
    s2 = np.array([0.192, 0.1078, 1])
    r, R, t = calculate_point(s1, s2, r1, r2, t1, t2)
    
    print('\n\nValid orientations\nr: {}\nR: \n{}\nt: {}'.format(r, R, t))
    T = np.block([[R, t.reshape(3,1)],
                  [0,0,0,1]])
    
    plot([np.eye(4), T], [r])

    plt.show()
    
    
