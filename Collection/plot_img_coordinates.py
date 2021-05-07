import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)

def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def task_a(): # Transforms ro to rc via Tco. Returns rc (not homog)
    rc_h = np.zeros((4, 4))
    
    for i in range(len(ro_h)):
        rc_h[i] = Tco @ ro_h[i]
    
    return rc_h[:, :-1]


def task_b(): # Converts rc into normalizes imgage coordinates, s
    s = np.zeros((4,3))
    rc = task_a()
    for i in range(len(rc)):
        s[i] = 1/rc[i, 2] * rc[i]
    return s

def task_c(ro, s):
    plt.scatter(ro[:,0], ro[:,1], label='rc')
    plt.scatter(s[:,0], s[:,1], label='s')
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    Tco = np.block([[rotx(2*np.pi/3), np.array([[0, 0, 2]]).T],
                   [np.zeros(3), 1]])
    
    ro = np.array([[0,0,0],
                   [1,0,0],
                   [1,1,0],
                   [0,1,0]])    
    ro_h = np.block([ro, np.ones((4,1))])

    rc = task_a()
    for i in range(len(rc)):
        print('rc{}: '.format(i+1), rc[i])
    
    s = task_b()
    for i in range(len(rc)):
        print('s{}: '.format(i+1), s[i])
    
    task_c(ro, s)

    
    
