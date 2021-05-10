import numpy as np


def skewm(r):
	# The skewm symmetric form of a vector
	S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
	return S

def expso3(u):
    R = np.identity(3) + np.sinc(np.linalg.norm(u)/np.pi)*skewm(u) + 0.5*(np.sinc(np.linalg.norm(u)/(2*np.pi)))**2 * skewm(u) @ skewm(u)
    return R

def Rt2T(R,t):
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def rc2H(x,xp):
    # Homography H = [r1,r2,t] from point is a plane
    A1 = np.hstack([x[0,0]*skewm(xp[:,0]), x[1,0]*skewm(xp[:,0]), skewm(xp[:,0])])
    A2 = np.hstack([x[0,1]*skewm(xp[:,1]), x[1,1]*skewm(xp[:,1]), skewm(xp[:,1])])
    A3 = np.hstack([x[0,2]*skewm(xp[:,2]), x[1,2]*skewm(xp[:,2]), skewm(xp[:,2])])
    A4 = np.hstack([x[0,3]*skewm(xp[:,3]), x[1,3]*skewm(xp[:,3]), skewm(xp[:,3])])
    A = np.vstack((A1, A2, A3, A4))
    # Singlar value decomposition
    _, _, V = np.linalg.svd(A);
    # Nullspace solution
    scale = np.sign(V[8,8])*np.linalg.norm(V[8,0:3])
    h1 = V[8,0:3].reshape(3,1)
    h2 = V[8,3:6].reshape(3,1)
    h3 = V[8,6:9].reshape(3,1)
    H = np.block([h1, h2, h3])/scale
    return H

def Hom2B(H1, H2, H3):
    # Calculate B = inv(K*K') from the homographies H1, H2, H3 of three planes
    # where K is the camera calibration matrix
    # Matrix rows
    A1, A2 = calc_h1h2_rows(H1[:,0],H1[:,1])
    A3, A4 = calc_h1h2_rows(H2[:,0],H2[:,1])
    A5, A6 = calc_h1h2_rows(H3[:,0],H3[:,1])
    # A7 = [0 1 0 0 0 0]'; % B(1,2) = B(2,1) = 0;
    # A8 = [1 0 -1 0 0 0]'; % B(1,1) = B(2,2);
    A = np.vstack((A1, A2, A3, A4, A5, A6))
    _, _,Vt = np.linalg.svd(A)
    V = Vt.T
    B = np.zeros((3,3))
    B[0,0] = V[0,5];
    B[0,1] = V[1,5];
    B[1,1] = V[2,5];
    B[0,2] = V[3,5];
    B[1,2] = V[4,5];
    B[2,2] = V[5,5];
    B[1,0] = B[0,1];
    B[2,0] = B[0,2];
    B[2,1] = B[1,2];
    
    B = B/B[2,2];
    return B


def calc_h1h2_rows(u,v):
    # Calculate constraint rows A1' and A2' for single view homography 
    # H of plane from the original contraints h1'*B*h2 = 0
    # h1'*B*h1 - h2'*B*h2 = 0 where B = inv(K*K')
    # h1 = H(:,1), h2 = H(:,2)
    # The matrix 
    # B = [b1 b2 b4; 
    #      b2 b3 b5;
    #      b4 b5 b6]
    # is stacked in the array b = [b1 b2 ... b6]
    # New constraints:
    # A1*b = 0 and A2*b = 0

    A1 = [u[0]*v[0],  
          u[0]*v[1]+u[1]*v[0],
          u[1]*v[1], 
          u[0]*v[2]+u[2]*v[0],
          u[1]*v[2]+u[2]*v[1], 
          u[2]*v[2]]
    A2 = [u[0]*u[0]-v[0]*v[0], 
          2*u[0]*u[1]-2*v[0]*v[1],
          u[1]*u[1]-v[1]*v[1], 
          2*u[0]*u[2]-2*v[0]*v[2],
          2*u[1]*u[2]-2*v[1]*v[2], 
          u[2]*u[2]-v[2]*v[2]]
    return A1, A2

def points2pixels(r,Tco,K):
    # Homogeneous parameters in camera frame
    rc = Tco @ r
    c = np.zeros((3,4))
    # Sensor readings in pixel coordinaes
    for i in range (0, 4):
        c[:,i] = K @ rc[0:3:,i]/rc[2,i] 
    return c

#Calculate intrinsic camera matrix K from a single view of three planes 
#using the homographies of the three planes. 
#Reference: Siciliano Section 10.5
#Ma, Soatto, Kosecka, Sastry Section 6.5.3. 
#Solution from Hartley and Zisserman, Algorithm 8.2 p. 225.

if __name__ == '__main__':
    # 4 Corners of a square
    r = np.array([[0, 0, 0, 1], [0.1, 0, 0, 1], [0.1, 0.1, 0, 1], [0, 0.1, 0, 1]])
    r = r.T
    # Camera parameter matrix
    K = np.array([[750, 0, 640], [0, 750, 512], [0, 0, 1]]) 
    ex = np.array([1, 0, 0]); ey = np.array([0, 1, 0]); ez = np.array([0, 0, 1]) 
    
    # Plane 1
    Rco = np.identity(3)
    ococ = np.array([0, 0, 0.5]) 
    Tco = Rt2T(Rco,ococ)
    c1 = points2pixels(r,Tco,K);
    # Plane 2
    Rco = expso3(0*ez) @ expso3(np.pi/4*ex) @ expso3(np.pi/4*ey)
    ococ = np.array([0, 0, 0.5]) 
    Tco = Rt2T(Rco,ococ)
    c2 = points2pixels(r,Tco,K);
    #Plane 3
    Rco = expso3(np.pi/3*ez) @ expso3(-np.pi/3*ex) @ expso3(-np.pi/3*ey) 
    ococ = np.array([0, 0, 0.5]) 
    Tco = Rt2T(Rco,ococ)
    c3 = points2pixels(r,Tco,K);

    # Given r and c1,c2,c3, calculate the homographies for the 3 planes
    H1 = rc2H(r,c1)
    H2 = rc2H(r,c2)
    H3 = rc2H(r,c3)
    
    # Calculate B = inv(K*K')
    B = Hom2B(H1, H2, H3)
    
    # Solve B = Kinv'*Kinv where K is upper diagonal
    K = np.linalg.inv(np.linalg.cholesky(B))
    
    # Normalize
    K = K.transpose()/K[2,2]

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print('Estiamted kamera parameter matrix:')
    print(K)
