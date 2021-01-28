import numpy as np
import matplotlib.pyplot as plt

def line2y(line, start, end):
    '''
    Takes a line on the form [a, b, c] and returns x and y (=ax+b) on the intervall [start, end]
    '''
    x = np.array([start, end])
    y = -1/line[1] * (line[0] * x + line[2])
    return x, y

def plot_hline(lines, start, end):
    x = np.zeros((len(lines), 2))
    y = np.zeros((len(lines), 2))
    for i in range(len(lines)):
        x[i], y[i] = line2y(lines[i], start, end)
        plt.plot(x[i], y[i], label=r'$l_{}$'.format(i))
    plt.grid(ls='--')
    plt.legend()
    plt.show()

def dist_line2point(line, point):
    '''
    Takes line on the form [a, b, c] and a homogenous point on the form [x, y, 1] and returns shortest dist between them
    '''
    n = line[:2]
    return abs(np.dot(line, point)/np.norm(n))

def intersect_lines(l1, l2):
    '''
    Takes two lines on the form [a, b, c] and return the intersection point on the form [x, y]
    '''
    return np.cross(l1, l2)[:2]
    
def line_through_2points(x1, x2):
    '''
    Takes two homogenous points on the form [x, y, 1] and returns a line [a, b, c] that intersect both points
    '''
    return np.cross(x1, x2)

def task1():
    l1 = np.array([1, -1, 0])
    l2 = np.array([0, 1, -1])
    plot_hline([l1, l2], -3, 3)
    
    
if __name__ == '__main__':
    # task1()
    print(line_through_2points(np.array([1,0,0]), np.array([0,1,1])))