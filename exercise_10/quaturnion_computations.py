import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def skew(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def weks(M): # From skew symetric matrix to vector
    if M[0,1] != -M[1,0] or M[0,2] != -M[2,0] or M[1,2] != -M[2,1]:
        print('ERROR: Matrix is not skew symetric')
    return np.array([M[2,1], M[0,2], M[1,0]])

def rot2log(R, margin=0.000001):
    '''
    Returns the exponential description, u=θk, of the rotation-matrix R
    '''
    eh = 0.5*(R-R.T) # eh = sin(theta)k^skew
    en = np.linalg.norm(weks(eh)) # en = |sin(theta)|
    if en < margin:
        g = 1 + (en**2)/6
    else:
        g = (np.arcsin(en)/en)
    return weks(g*eh)

def rot2quat(R): # Limited to θ<=pi/2
    u = rot2log(R)
    t, k = decompose_exp(u)
    return np.block([np.cos(t/2), k*np.sin(t/2)])

def exp2rot(u):
    '''
    Returns the rotation matrix from the exponential description u=θk
    '''
    S = skew(u)
    un = np.linalg.norm(u)
    return np.identity(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

def exp2quat(u):
    '''
    Returns the quaturnion [r, i, j, k] from the exponential description u=θk
    '''
    phi = np.linalg.norm(u)
    return np.block([np.cos(phi), np.sinc(phi/np.pi)*u])

def q_from_k_theta(k,theta):
    return np.block([np.cos(theta/2), np.sin(theta/2)*k])

def quat2rot(q): # Limited to θ<=pi/2
    '''
    Returns the rotation-matrix corresponding to the quaturnion q on the form [r, i, j, k]
    '''
    q_norm = make_unit_quat(q)
    s = q_norm[0]
    v = q_norm[1:4]
    V = skew(v)
    return np.eye(3) + 2*s*V + 2*V@V


def quat2log(q):
    '''
    Returns the logarithm u=θk corresponding to the quaturnion q on the form[r, i, j, k]
    '''
    q_norm = make_unit_quat(q)
    sigma = q_norm[1:]
    return (np.arcsin(np.linalg.norm(sigma)) * sigma) / np.linalg.norm(sigma)

def make_unit_quat(q): # Returns the corresponding unit quaturnion
    return q/q_magn(q)

def cay(u): # Cayley transform. Used as R = cay(roh) where roh is a rodrigues vector
    return (np.eye(3)+skew(u))@np.linalg.inv(np.eye(3)-skew(u))

def rodrigues_vector(R): # Returns a rodrigues vector corresponding to rotation matrix R
    return (1/(np.trace(R)+1)) * np.array([(R[2,1]-R[1,2]), (R[0,2]-R[2,0]), (R[0,2]-R[2,0])])

def decompose_exp(u): # takes a vector u=θk and returns θ and k
    return np.linalg.norm(u), (1/np.linalg.norm(u))*u

def exp_combine(theta, k):
    return theta * k





def q_prod(q1,q2):
    '''
    Returns the product of quaturnions q1, q2
    '''
    if q1.shape[0] == 4:
        s1 = q1[0]; v1 = q1[1:4]
    else:
        s1 = 0; v1 = q1[0:3]
    if q2.shape[0] == 4:
        s2 = q2[0]; v2 = q2[1:4]
    else:
        s2 = 0; v2 = q2[0:3]
    s = s1*s2 - np.dot(v1,v2)
    v = s1*v2 + s2*v1 + np.cross(v1,v2)
    return np.block([s, v])

def q_magn(q):
    '''
    Returns the magnitude of a quaturnion
    '''
    return np.sqrt(np.inner(q,q))

def q_conj(q):
    '''
    Returns the conjugate of a quaturnion 
    '''
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_inv(q):
    '''
    Returns the inverse of a quaturnion
    '''
    return q_conj(q)/q_magn(q)**2

def QL(q):
    s = q[0]; v = q[1:4]
    return np.block([[s, -v], [v.reshape(3,1), s*np.eye(3) + skew(v)] ] )

def QR(q):
    s = q[0]; v = q[1:4]
    return np.block([[s, -v], [v.reshape(3,1), s*np.eye(3) - skew(v)] ] )

def quat_rotation(q,a):
    return q_prod(q_prod(q,a),q_conj(q))[1:4]

def shepperd(R):
    z00 = np.trace(R)
    z11 = R[0,0] + R[0,0] - z00
    z22 = R[1,1] + R[1,1] - z00
    z33 = R[2,2] + R[2,2] - z00
    #Find a large zii to avoid division by zero
    if z00 >= 0.5:
        w = np.sqrt(1.0 + z00)
        wInv = 1.0/w
        x = (R[2,1] - R[1,2])*wInv
        y = (R[0,2] - R[2,0])*wInv
        z = (R[1,0] - R[0,1])*wInv
    elif z11 >= 0.5:
        x = np.sqrt(1.0 + z11);
        xInv = 1.0/x
        w = (R[2,1] - R[1,2])*xInv
        y = (R[1,0] + R[0,1])*xInv
        z = (R[2,0] + R[0,2])*xInv
    elif z22 >= 0.5:
        y = np.sqrt(1.0 + z22)
        yInv = 1.0/y
        w = (R[0,2] - R[2,0])*yInv
        x = (R[1,0] + R[0,1])*yInv
        z = (R[2,1] + R[1,2])*yInv
    else:
        z = np.sqrt(1.0 + z33)
        zInv = 1.0/z
        w = (R[1,0] - R[0,1])*zInv
        x = (R[2,0] + R[0,2])*zInv
        y = (R[2,1] + R[1,2])*zInv
    s = 0.5*w
    v = 0.5*np.array([x, y, z])
    if s < 0:
        s = - s
        v = - v
    return np.block([s, v])


if __name__ == "__main__":
    t, k = np.pi/2, np.array([1,0,0])
    t, k = np.pi/3, np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    u = exp_combine(t, k)
    print(u)
    print(rot2log(exp2rot(u)))
    print(quat2log(exp2quat(u)))

    R = exp2rot(u)
    print('\n{}'.format(R.flatten()))
    print(exp2rot(rot2log(R)).flatten())
    print(quat2rot(rot2quat(R)).flatten())
    
    q = exp2quat(u)
    print('\n{}'.format(q))
    print(exp2quat(quat2log(q)))
    print(rot2quat(quat2rot(q)))
