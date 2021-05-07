import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def exp2(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],\
                      [np.sin(theta), np.cos(theta)]]) 

def Ese2(theta):
    a = np.sinc(theta/np.pi)
    b = (theta/2) * (np.sinc(theta/(2*np.pi)))**2
    return np.array([[a, -b], [b, a]])

def Ese2_inv(theta):
    a = theta/2
    if a > 0.0001:
        b = a/np.tan(a)
    else:
        b = 1 - a**2/3 - a**4/45
    return np.array([[b, a], [-a, b]])

def Fse2(theta):
    if np.abs(theta) > 0.00001:
        a = (theta - np.sin(theta))/np.square(theta)
    else:
        a = theta/6 - theta**3/120
    b = (1/2) * (np.sinc(theta/(2*np.pi)))**2
    return np.array([[a, b], [-b, a]])

def jr2(theta,rho):
    E = Ese2(theta)
    F = Fse2(theta)
    return np.block([[np.array([1, 0, 0])],
                      [(F.T @ rho).reshape(2,1),E.T]])

def jr2_inv(theta,rho):
    Einv = Ese2_inv(theta)
    F = Fse2(theta)
    return np.block([[np.array([1, 0, 0])],
            [-(Einv.T @ F.T @ rho).reshape(2,1), Einv.T]])  
    
def integrate_se2(h, x, v, om):
    return np.block([x[0] + h*om, x[1:3] + exp2(x[0]) @ Ese2(om*h) @ (v*h)])

H = 1; ex = np.array([1, 0]); ey = np.array([0,1])
om = np.pi/2; vr = ex

#Exact exponential solution for interval with constant om and v
R = exp2(H*om)
p = Ese2(H*om) @ (H*vr)
print(R, '\n', p)

## Geometric integration
#Rj = np.eye(2)
#N = 3
#h = H/N
#p_g = np.zeros((2,N+1))
#for i in range(1, N+1):
#    p_g[:,i]= p_g[:,i-1] + Rj @ Ese2(om*h) @ (vr*h)
#    Rj = Rj @ exp2(om*h)
#print(p_g[:,N])

# Geometric integration
Ri = np.eye(2)
N = 2
h = H/N
X = np.zeros((3,N+1))
p_g = np.zeros((2,N+1))
for i in range(1, N+1):
    X[:,i] = integrate_se2(h, X[:,i-1], vr, om)
    p_g[:,i] = X[1:3,i]
print(p_g[:,N])

# Euler integration
Ri = np.eye(2)
N = 1000
h = H/N
p_e = np.zeros((2,N+1))
for i in range(1, N+1):
    p_e[:,i]= p_e[:,i-1] + Ri @ (vr*h)
    Ri = Ri @ exp2(om*h)
print(p_e[:,N])

#Integration of logarithm vector
N = 10
h = H/N
u = np.zeros((3,N+1))
for i in range(1, N+1):
    tt = u[0,i-1]
    rr = u[1:3,i-1]
    Om = np.block([om,vr])
    u[:,i]= u[:,i-1] + jr2_inv(tt,rr) @ (Om*h)
thetaN = u[0,N]
p_u = Ese2(thetaN) @ u[1:3,N]
print(p_u)

plt.figure(1)
plt.figure(1).clear()
plt.plot(p_e[0], p_e[1], p_g[0], p_g[1], p[0], p[1], 'ro',p_u[0], p_u[1], 'gd')
plt.xlabel('x')
plt.legend(['Euler', 'Exponential', 'Exact', 'Log'], shadow=True)
plt.title('Integration in in SE(2)')
plt.show()


alpha = np.pi/6; rho = np.array([0.5, 0.2])
print(jr2(alpha,rho) @ jr2_inv(alpha,rho))