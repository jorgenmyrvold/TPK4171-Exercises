import numpy as np
np.set_printoptions(suppress=True, precision=8)

def skew(M):
    return np.array([[0, -M[2], M[1]],
                     [M[2], 0, -M[0]],
                     [-M[1], M[0], 0]])


    
def jacobian(u, left, margin=0.00001):
    if not left: u = -u
    theta = np.linalg.norm(u)
    uh = skew(u)
    
    if theta > margin:
        a = (1-np.cos(theta))/(theta**2)
        b = (theta - np.sin(theta))/(theta**3)
    else:
        a = 0.5 - theta**2/24
        b = 1/3 - theta**2/120
    return np.eye(3) + a*uh + b*uh@uh


def jacobian_inv(u, left, margin=0.00001):
    if not left: u = -u
    theta = np.linalg.norm(u)
    theta_half = theta/2
    uh = skew(u)
    
    if theta > margin:
        a = (1 - theta_half/np.tan(theta_half))/(theta**2)
    else:
        a = 1/12 - 1/180*theta**2
    return np.eye(3) - 0.5*uh + a*uh@uh
    
    
    
if __name__ == "__main__":
    k = np.array([1,0,0])
    t = np.pi/4
    u = t * k
    left_j = jacobian(u, left=True)
    right_j = jacobian(u, left=False)
    left_j_inv = jacobian_inv(u, left=True)
    right_j_inv = jacobian_inv(u, left=False)

    print('Left jacobian: \n{}\n\nRight jacobian: \n{}'. format(left_j, right_j))
    print('\nLeft inverse jacobian: \n{}\n\nRight inverse jacbian: \n{}'.format(left_j_inv, right_j_inv))
    
    print("test:\n{}\n{}".format(left_j@left_j_inv, right_j@right_j_inv))