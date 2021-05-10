import numpy as np
np.set_printoptions(suppress=True, precision=4)


# 1. Line from two points
def points2line(x, y):  # Eq. 400
    return np.array([x[3]*y[:3] - y[3]*x[:3], np.cross(x[:3], y[:3])])


# 2. Plane from three points
def points2plane(x, y, z): # Algorithm in chapter 9.11
    sub1 = np.block([[x[1:]], [y[1:]], [z[1:]]])
    sub2 = np.block([[x[0], x[2:]], [y[0], y[2:]], [z[0], z[2:]]])
    sub3 = np.block([[x[:2], x[3]], [y[:2], y[3]], [z[:2], z[3]]])
    sub4 = np.block([[x[:3]], [y[:3]], [z[:3]]])
    
    D1 = np.linalg.det(sub1)
    D2 = np.linalg.det(sub2)
    D3 = np.linalg.det(sub3)
    D4 = np.linalg.det(sub4)
    
    plane = np.array([-D1, -D2, -D3, -D4])
    return plane


# 3. Plane from line and one point
# This or points2plane is wrong
def line_and_point2plane(l, x): # Eq. 427
    return np.array(np.block([-x[3]*l[1] + np.cross(x[:3], l[0]), np.dot(x[:3], l[1])]))


# 4. Line from intersection of two planes
def planes2line(u, v): # Eq. 450
    return np.array([np.cross(u[:3], v[:3]), u[3]*v[:3] - v[3]*u[:3]])


# 5. Point from intersection of line and plane
def line_and_plane2point(l, u):  # Script chapter 9.20
    x = np.array(np.block([np.dot(-u[3], l[0]) + np.cross(u[:3], l[1]), np.dot(u[:3], l[0])]))
    if x[3] == 0:
        print('Line paralell to plane, intersection at infinity')
    return x


# 6. Point from intersection of two lines
def lines2point(l1, l2): # Script chapter 9.24
    if (np.dot(l1[0], l2[1]) + np.dot(l1[1], l2[0])) != 0:
        print('Lines do not intersect')
        return
    
    n = np.cross(l1[0], l2[0])
    v = np.array(np.block([np.cross(n, l1[0]), np.dot(n, l1[1])]))
    x = np.array(np.block([-v[3]*l2[0] + np.cross(v[:3], l2[1]), np.dot(v[:3], l2[0])]))
    return x


# 7. Point from intersection of three planes
def planes2point(u, v, w): # Script chapter 9.22
    l = planes2line(u, v)
    x = np.array(np.block([-w[3]*l[0] + np.cross(w[:3], l[1]), np.dot(w[:3], l[0])])) # Eq. 479
    return x


# 8. Check if a point is on a plane
def is_point_on_plane(x, u):  # Script chapter 9.22
    return np.dot(x[:3], u[:3]) + x[3]*u[3] == 0


# 9. Check if a line is in a plane
def is_line_in_plane(l, u): # Eq. 476
    if np.dot(u[:3], l[0]) != 0: return False
    if (-u[3]*l[0] + np.cross(u[:3], l[1]))!= np.zeros(3): return False
    return True


# 10. Distance between two lines
def dist_between_lines(l1, l2): # Script s. 117
    ln = np.array([np.cross(l1[0], l2[0]), np.cross(l1[0], l2[1]) + np.cross(l1[1], l2[0])]) # Eq. 505
    u = np.array(np.block([np.cross(l1[0], ln[0]), np.dot(l1[0], ln[1])]))
    v = np.array(np.block([np.cross(l2[0], ln[0]), np.dot(l2[0], ln[1])]))
    x = np.array(np.block([-v[3]*l1[0] + np.cross(v[:3], l1[1]), np.dot(v[:3], l1[0])]))
    y = np.array(np.block([-u[3]*l2[0] + np.cross(u[:3], l2[1]), np.dot(u[:3], l2[0])]))
    return np.linalg.norm((y[3]*x[:3] - x[3]*y[:3]) / (x[3]*y[3]))


# 11. Distance from point to line
def dist_point2line(x, l):
    return np.linalg.norm(np.cross(l, (-x[3]*l[1] + np.cross(x[:3], l[0])) / (x[3]*np.dot(l[0],l[0]))))



if __name__ == "__main__":
    # Create a line from two points
    x = np.array([1,2,3,1])
    y = np.array([4,3,2,1])
    print("\n1. Line defines by x and y is \n(l, l'): \n{}".format(points2line(x, y)))
    
    
    # Create a plane from three points
    x = np.array([1,2,3,1])
    y = np.array([4,3,2,1])
    z = np.array([2,2,2,1])
    print('\n2. Plane defined by points x, y and z are: \nu: {}'.format(points2plane(x, y, z)))
    
    
    # Create a plane from a line and two points
    # This should probably be the same as in task 2, but i'm not sure which one is correct...
    x = np.array([1,2,3,1])
    y = np.array([4,3,2,1])
    l = points2line(x,y)
    z = np.array([2,2,2,1])
    print('\n3. Plane defined by line l and point x is: \nu: {}'.format(line_and_point2plane(l, z)))
    
    
    #Line from intersection of two planes
    u = np.array([1,2,3,1])
    v = np.array([4,3,2,1])
    l = planes2line(u, v)
    print("\n4. Line defines by the planes u and v is: \n(l, l'): \n{}".format(l))
    
    
    # Point of intesection between line and plane
    l = np.array([[0,0,1], np.cross([1,1,0],[0,0,1])])
    u = np.array([1,1,1, np.sqrt(3)])
    x = line_and_plane2point(l, u)
    print("\n5. Point, x, at intersection between line l and plane u \nx: {}".format(x))
    # Test Expected zeros
    print('-- Testing: Expecting zeros: --')
    print('test_u ', np.dot(x[:3], u[:3]) + x[3]*u[3])
    print('test_Ls', np.dot(x[:3], l[1]))
    print('test_Lv', -x[3]*l[1] + np.cross(x[:3], l[0]))
    
    
    # Intersection of two lines
    x = np.array([1,2,3,1])
    l1 = np.array([[1,0,0], np.cross(x[:3], [1,0,0])])
    l2 = np.array([[0,0,1], np.cross(x[:3], [0,0,1])])
    print('\n6. Homogenous point, x, at intersection of l1 and l2: \nx: {}'.format(lines2point(l1, l2)))


    # Intersection of three planes
    u = np.array([1, 0, 0, 1])
    v = np.array([0, 1, 0, 2])
    w = np.array([0, 0, 1, 2])
    x = planes2point(u, v, w)
    print("\n7. Intersection of three planes u, v, w: \n{}".format(x))
    # Test if x is on all planes
    print('x is on u:', is_point_on_plane(x, u)) # Task 8
    print('x is on v:', is_point_on_plane(x, v))
    print('x is on w:', is_point_on_plane(x, w))
    
    
    # Distance between two lines
    l1 = np.array([[1,0,0], [0,0,0]])
    l2 = np.array([[0,1,0], [-1,0,1]])
    print('\n10. Distance between l1 and l2: {}'.format(dist_between_lines(l1, l2)))
    
    
    # Distance from point to plane
    l = np.array([[0,1,0], np.cross([1,0,1], [0,1,0])])
    x = np.array([0,2,4,2])
    print('\n11. Distance between point x to line l: {}'.format(dist_point2line(x, l)))