import numpy as np
from numpy.linalg import inv
from math import atan2, cos, sin, pi
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from linear import DemoLinearCode
from nonlinear import DemoNonlinearCode
    
# main.py
elemSet = np.array([[0, 1], 
                    [1, 2],
                    [2, 3], 
                    [3, 4],
                    [4, 5]])
                    
nodeSet = np.array([[ 0, 0],
                    [20, 0],
                    [40, 0 ],
                    [60, 0], 
                    [80, 0],
                    [100, 0]])
                    
B = 5*np.ones((len(elemSet)))  
H = 5*np.ones((len(elemSet)))
Y = 200e2*np.ones((len(elemSet)))  

f = np.array([[5, 2, 500]])

bc = np.array([[0, 1, 0],
               [0, 2, 0], 
               [0, 3, 0]])
               
inc = 20  

u, Ri, alpha = DemoNonlinearCode(nodeSet, elemSet, Y, H, B, f, bc, inc, 'on') 
# ul, Ri = DemoLinearCode(nodeSet, elemSet, Y, H, B, f, bc, inc, 'on')

nElem = elemSet.shape[0]
nNode = nodeSet.shape[0]

# Plot Linear Deformation  
fig1 = plt.figure(1)

# for i in range(nElem):
#     m = elemSet[i,0] 
#     n = elemSet[i,1]
    
#     plt.plot(nodeSet[[m,n],0], nodeSet[[m,n],1], 'k-', linewidth=2)

#     plt.plot(nodeSet[[m,n],0] + [ul[m,0], ul[n,0]], 
#              nodeSet[[m,n],1] + [ul[m+nNode,0], ul[n+nNode,0]],
#              'b-', linewidth=2)
             
#     plt.axis('equal')
#     plt.hold(True)

for j in range(inc):

    print(u[:, j])
    
    for i in range(nElem):
    
        m = elemSet[i,0]
        n = elemSet[i,1]

        plt.plot(
            [nodeSet[m, 0], nodeSet[n, 0]], 
            [nodeSet[m, 1], nodeSet[n, 1]],
            'k-', 
            linewidth=2
        ) 

        plt.plot(
            [nodeSet[m, 0]+u[m, j], nodeSet[n, 0]+u[n, j]], 
            [nodeSet[m, 1]+u[m+nNode,j], nodeSet[n, 1]+u[n+nNode,j]],
            'r-', 
            linewidth=2
        ) 
                 
        # plt.axis('equal')

  
plt.show()