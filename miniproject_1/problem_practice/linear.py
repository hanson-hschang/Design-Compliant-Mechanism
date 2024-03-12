import numpy as np
from numpy.linalg import inv
from math import atan2, cos, sin, pi
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

# DemoLinearCode.py
def DemoLinearCode(nodeSet, elemSet, Y, H, B, F, bc, numSubstep, cmdout):
    
    nDOF = 3 
    nNode = len(nodeSet[:,0])
    totalDOF = nDOF*nNode
    nElem = len(elemSet[:,0])
    
    BC = np.full((nNode*3,1), np.nan)
    for i in range(len(bc[:,0])):
        BC[bc[i,0] + nNode*(bc[i,1]-1)] = bc[i,2]
        
    f = np.zeros((nNode*3,1))
    for i in range(len(F[:,0])):
        f[F[i,0] + nNode*(F[i,1]-1)] = F[i,2]
        
    BC = BC.flatten()
    
    Kglobal = getStiffMatrix(nodeSet, elemSet, Y, H, B)
    
    K = Kglobal.copy()
    bcwt = np.mean(np.diag(K))
    
    freeBC = np.isnan(BC)
    cnstBC = ~freeBC
    
    f[freeBC] = f[freeBC] - K[freeBC,:][:,cnstBC]@BC[cnstBC]
    
    K[cnstBC,:] = 0
    K[:,cnstBC] = 0 
    K[cnstBC,cnstBC] = bcwt*np.eye(len(BC[cnstBC]))
        
    f[cnstBC] = bcwt*BC[cnstBC]
    
    u = inv(K)@f 
    f = Kglobal@u
    
    return u, f, Kglobal

# getStiffMatrix
def getStiffMatrix(nodeSet, elemSet, Y, H, B):

    nElem = len(elemSet[:,0])
    nNode = len(nodeSet[:,0])
    nDOF = 3
        
    GM = csc_matrix((nNode*nDOF, nNode*nDOF))
    
    for i in range(nElem):
       
        sctr = elemSet[i,:] 
        sctrVct = [sctr[0], sctr[0]+nNode, sctr[0]+2*nNode]
       
        EMatrix = getEMatrixStiff(elemSet[i,:], nodeSet, Y, H, B)
       
        EMatrixTrans = getEMatrixTrans(elemSet[i,:], nodeSet)
        
        GM[np.ix_(sctrVct, sctrVct)] += EMatrixTrans.T @ EMatrix @ EMatrixTrans
        
    return GM

# getEMatrixStiff
def getEMatrixStiff(elemSet, nodeSet, Y, H, B):

    nElem = len(elemSet[:,0])
    
    Ke = np.zeros((6,6,nElem))
    
    for i in range(nElem):
       
        in_ = elemSet[i,0] 
        jn = elemSet[i,1]
        E = Y[i]  
        A = B[i]*H[i]   
        I = B[i]*H[i]**3/12     
        L = np.linalg.norm(nodeSet[jn,:] - nodeSet[in_,:])
        L2 = L**2
        
        IdL = 6*I/L 
        IdL2 = 12*I/L2  
        I_4 = 4*I 
        I_2 = 2*I
        
        Ke[:,:,i] = (E/L)*np.array([
            [A,   -A,    0,       0,       0,      0],
            [-A,   A,    0,       0,       0 ,     0], 
            [0,    0,    IdL2,   -IdL2,    IdL,    IdL],
            [0,    0,   -IdL2,    IdL2,   -IdL,   -IdL],
            [0,    0,    IdL,    -IdL,     I_4,    I_2], 
            [0,    0,    IdL,    -IdL,     I_2,    I_4]])
        
    return Ke
    
# getEMatrixTrans    
def getEMatrixTrans(elemSet, nodeSet):

    nElem = len(elemSet[:,0])
    
    Te = np.zeros((6,6,nElem))
    
    for i in range(nElem):
       
        in_ = elemSet[i,0]
        jn = elemSet[i,1]
        
        theta = atan2(nodeSet[jn,1] - nodeSet[in_,1], nodeSet[jn,0] - nodeSet[in_,0])
        Ecos = cos(theta)
        Esin = sin(theta)
        
        Te[:,:,i] = np.array([
            [Ecos,   0,      Esin,  0,      0,   0],
            [0,      Ecos,   0,     Esin,  0,   0],
            [-Esin,  0,      Ecos, 0,       0,   0],
            [0,     -Esin,   0,   Ecos,    0,   0],
            [0,      0,      0,      0,     1,   0],
            [0,      0,      0,      0,     0,   1]])
        
    return Te