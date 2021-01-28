import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)


def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])
    
def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def line_through_2points(x1, x2):
    '''
    Takes two homogenous points on the form [x, y, 1] and returns a line [a, b, c] that intersect both points
    '''
    return np.cross(x1, x2)

def intersect_lines(l1, l2):
    '''
    Takes two lines on the form [a, b, c] and return the intersection point on the form [x, y]
    '''
    return np.cross(l1, l2)

def task_a():
    lines = np.zeros((4, 3))
    for i in range(len(lines)):
        lines[i] = line_through_2points(x[i], x[(i+1)%4])

    y1 = np.cross(lines[0], lines[2]) # Find intersection
    y2 = np.cross(lines[1], lines[3])
    
    print(lines)
    print('y1:', y1, '\ny2:', y2)
    
def task_b():
    s = np.zeros((4,3))    
    rc_h = np.zeros((4,4))
    ro_h = np.block([ro, np.ones((4,1))])
    
    for i in range(len(ro_h)):
        rc_h[i] = Tco @ ro_h[i]
        s[i] = rc_h[i, :3] / rc_h[i, 2]
        print('s{}: '.format(i+1), s[i]) 
    
    return s     
        
def task_c(s):
    lines_h = np.zeros((4,3))
    for i in range(len(s)):
        lines_h[i] = np.cross(s[i], s[(i+1)%4])
        print('lambda{}: '.format(i+1), lines_h[i]) 
    return lines_h

def task_d(lines_h):
    z1 = np.cross(lines_h[0], lines_h[2])  # Find intersection
    z2 = np.cross(lines_h[1], lines_h[3])
    print('\nz1:', z1, '\nz2:', z2)
    return z1, z2

def task_e(z1, z2):
    print('\nlambda_z:', np.cross(z1, z2))  # Find line through z1 and z2
    
    

if __name__ == "__main__":
    t = np.array([[0,0,2]]).T
    Tco = np.block([[rotx(np.deg2rad(120)) @ rotz(np.deg2rad(45)), t], 
                    [np.zeros(3), 1]])
    
    ro1 = np.array([0,0,0])
    ro2 = np.array([1,0,0])
    ro3 = np.array([1,1,0])
    ro4 = np.array([0,1,0])
    ro = np.array([ro1, ro2, ro3, ro4])
    
    x1 = np.array([[0,0,1]])
    x2 = np.array([[1,0,1]])
    x3 = np.array([[1,1,1]])
    x4 = np.array([[0,1,1]])
    x = np.array([x1, x2, x3, x4])
    
    task_a()
    # Points y1 and y2 are points at infinity 
    
    s = task_b()
    
    lines_h = task_c(s)
    
    z1, z2 = task_d(lines_h)
    # They corresponds to the points at infinity, y1 and y2 in the xy plane
    
    task_e(z1, z2)
    # This is the horizon
    
    
    
    