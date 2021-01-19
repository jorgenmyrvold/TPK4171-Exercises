import numpy as np
import matplotlib.pyplot as plt
from utils import *
np.set_printoptions(precision=4, suppress=True)


def hough_transform():
    x = np.array([0, 1, 2, 3, 0])
    y = np.array([0, 0, 0, 0, 1])
    theta = np.linspace(0, 2*np.pi, 100)
    
    for i in range(len(x)):
        plt.plot(theta, x[i]*np.sin(theta) + y[i]*np.cos(theta), label='i={}'.format(i+1))
    
    # From the figure we find that the intersection is where
    theta_intersect = np.pi
    rho_intersect = 0
    
    # This parameterization corresponds to the line
    y_reg = np.zeros(100)
    
    x_reg = np.linspace(0, 6.5, 100)
    plt.plot(x_reg, y_reg, label='Regression', color='c', linestyle='-.')
    
    plt.scatter(x,y)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\rho$')
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    hough_transform()
    
    '''
    Ser fra plottet at linjene 1-4 krysser i samme punkt, når theta=pi og rho=0.
    
    Løsningen blir linjen y=0 som er merket i som turkis stiplet linje i plottet
    '''