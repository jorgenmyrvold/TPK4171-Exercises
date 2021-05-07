import numpy as np

def skewm(r):
    return np.array([[0,-r[2],r[1]], [r[2],0,-r[0]], [-r[1],r[0],0]])

def expso3(u):
    S = skewm(u); un = np.linalg.norm(u)
    return np.identity(3) + np.sinc(un/np.pi)*S + 0.5*(np.sinc(un/(2*np.pi)))**2 * S@S

def expquat(u):
    phi = np.linalg.norm(u)
    return np.block([np.cos(phi), np.sinc(phi/np.pi)*u])

def q_to_R(q):
    s = q[0]; v = q[1:4]
    V = skewm(v)
    return np.eye(3) + 2*s*V + 2*V@V

def q_prod(q1,q2):
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
    return np.sqrt(np.inner(q,q))

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_inv(q):
    return q_conj(q)/q_magn(q)**2

def QL(q):
    s = q[0]; v = q[1:4]
    return np.block([[s, -v], [v.reshape(3,1), s*np.eye(3) + skewm(v)] ] )

def QR(q):
    s = q[0]; v = q[1:4]
    return np.block([[s, -v], [v.reshape(3,1), s*np.eye(3) - skewm(v)] ] )

def quat_rotation(q,a):
    return q_prod(q_prod(q,a),q_conj(q))[1:4]

def q_from_k_theta(k,theta):
    return np.block([np.cos(theta/2), np.sin(theta/2)*k])


ex = np.array([1,0,0]); ey = np.array([0,1,0]); ez = np.array([0,0,1])
zz = np.zeros(3)
u1 = ex*np.pi/6; u2 = - ex*np.pi/3; u3 = (ex+ez)*ex*np.pi/12
u4 = ex*np.pi/2

q1 = expquat(u1/2)
q2 = expquat(u1/2)
q3 = expquat(u1/2)
q4 = expquat(u1/2)

R1 = q_to_R(q1)
R2 = q_to_R(q2)
R3 = q_to_R(q3)
R4 = q_to_R(q4);

q12 = q_prod(q1,q2)
R12 = R1@R2
print('R12 from q12 = \n {}\n R12 =\n {}'.format(q_to_R(q12), R12))
b = ex + 2*ey + 3*ez
aq = quat_rotation(q1,b)
aM = (QL(q1) @ QR(q_conj(q1)) @ np.block([0, b]))[1:4]
aR = R1 @ b
print(' q1 b q1* = \n {}\n QL(q) QR(q_conj(q)) b =\n {}\n R1 b =\n {}'.format(aq, aM, aR))
p = q1+q2+q3
pinv = q_inv(p)
print('\n p = {}\n pinv = {}\n p p_inv = {}\n '.format(p, pinv, q_prod(p, pinv)))