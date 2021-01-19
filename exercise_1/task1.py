import numpy as np
np.set_printoptions(precision=4, suppress=True)


k = 1500
u0 = 640
v0 = 512

K = np.array([[k, 0, u0],
              [0, k, v0],
              [0, 0, 1]])

K_inv = np.array([[1/k, 0, -u0/k],
                  [0, 1/k, -v0/k],
                  [0, 0, 1]])

def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def task_a():
    r = np.array([[0.1, 0.2, 0.5],
                  [1, 2, 5],
                  [0.1, 0.2, 1]])

    s = np.zeros((3,3))
    for i in range(len(r)):
        s[i] = r[i] / r[i, 2]
    
    p = np.zeros((3,3))
    for i in range(len(s)):
        p[i] = K @ s[i]
    
    for i in range(len(s)):
        print('s{}: '.format(i+1), s[i])
    for i in range(len(p)):
        print('p{}: '.format(i+1), p[i])

    
    
def task_b():
    p = np.array([[0, 0, 1],
                  [740, 612, 1],
                  [1280, 1024, 1]])
    
    s = np.zeros((3,3))
    for i in range(len(p)):
        s[i] = K_inv @ p[i]
    
    for i in range(len(s)):
        print('s{}: '.format(i+1), s[i])
    

def task_c():
    tco = np.array([[0,0,4]]).T
    Rco = rotx(np.pi/2)
    Tco = np.block([[Rco, tco],
                    [np.zeros(3), 1]])
    
    rop = np.array([[1,1,0]]).T
    rop_h = np.block([[rop],[1]])
    
    rcp_h = Tco @ rop_h
    
    C = 1/tco[2] * (K @ np.block([Rco, tco]))
    s = C @ rop_h
    p = K @ s
    print('C:\n', C)
    print('s:\n', s)
    print('p:\n', p)
    
    
    
if __name__ == '__main__':
    # task_a()   # Finished
    # task_b()   # Finished
    # task_c()   # Have answers, unsure if correct