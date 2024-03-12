
import numpy as np
from math import atan2, cos, sin, pi
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt

# DemoNonlinearCode.py
def DemoNonlinearCode(nodeSet, elemSet, Y, H, B, F, bc, numSubstep, cmdout):

    # initialize variables
    
    totDOF = len(nodeSet)*3  
    
    BC = np.full(len(nodeSet)*3, np.nan) 
    for i in range(len(bc)):
        BC[bc[i,0] + len(nodeSet)*(bc[i,1]-1)] = bc[i,2]
        
    f = np.zeros(len(nodeSet)*3)
    for i in range(len(F)): 
        f[F[i,0] + len(nodeSet)*(F[i,1]-1)] = F[i,2]

    Kglobal = getStiffMatrix(nodeSet, elemSet, Y, H, B)  
    
    K = Kglobal.copy()
    bcwt = np.mean(K.diagonal())
    
    freeBC = np.isnan(BC)
    cnstBC = ~freeBC
    
    f = externalLoadReduction(f, K, BC, freeBC, cnstBC, bcwt) 
    K = StiffnessReduction(K, cnstBC, BC, bcwt)
    
    uLinear = inv(K)@f
    Freaction = Kglobal@uLinear
    F = Freaction
    
    # initialize displacement, force, etc arrays
    
    u = np.zeros((totDOF,numSubstep)) 
    Ri = np.zeros((totDOF,numSubstep))
    lam = np.zeros(numSubstep)
    K = np.zeros((totDOF,totDOF,numSubstep))
    R = np.zeros((totDOF,numSubstep))
    
    alpha = np.zeros(len(elemSet))
    
    # load stepping loop
    
    for ss in range(numSubstep):
        
        lam[ss] = (ss+1)/numSubstep
         
        for iter in range(1000):
            
            # update variables
            result = getGTStiffness(elemSet, nodeSet, Y, H, B, u[:,ss], alpha)   
            
            
            K[:,:,ss] = result[0].toarray()
            Ri[:,ss] = result[1]
            alpha = result[2]
            R[:,ss] = f - Ri[:,ss]
            
            # check convergence
            
            convParam = np.linalg.norm(R[freeBC,ss])**2/(1+np.linalg.norm(f[freeBC])**2)
            
            if convParam <= 1e-8:
                break
            
            # reduce matrices
            
            bcwt = np.mean(np.diag(K[:,:,ss]))        
            Rred = externalLoadReduction(R[:,ss], K[:,:,ss], BC, freeBC, cnstBC, bcwt)
            Kred = StiffnessReduction(K[:,:,ss], cnstBC, BC, bcwt)
            
            # update displacement
            
            u[:,ss] += np.linalg.inv(Kred)@Rred
            
    return u, Ri, alpha
    
# Functions similar to Matlab versions

# StiffnessReduction
def StiffnessReduction(K, cnstBC, BC, bcwt):

    K[cnstBC, :] = 0
    K[:, cnstBC] = 0
    K[np.ix_(cnstBC, cnstBC)] = bcwt * np.eye(len(BC[cnstBC]))

    return K

# externalLoadReduction 
def externalLoadReduction(f, K, BC, freeBC, cnstBC, bcwt):

    f[freeBC] = f[freeBC] - K[np.ix_(freeBC, cnstBC)] @ BC[cnstBC]
    
    f[cnstBC] = bcwt * BC[cnstBC]
    
    return f

# getStiffMatrix
def getStiffMatrix(nodeSet, elemSet, Y, H, B):

    nElem = elemSet.shape[0]
    nNode = nodeSet.shape[0]
    nDOF = 3
        
    GM = csc_matrix((nNode*nDOF, nNode*nDOF))
    
    for i in range(nElem):
       
        sctr = elemSet[i,:]
        sctrVct = [sctr[0], sctr[1], sctr[0]+nNode, sctr[1]+nNode, sctr[0]+2*nNode, sctr[1]+2*nNode]
        
        EMatrix = getEMatrixStiff(elemSet[i,:], nodeSet, Y[i], H[i], B[i])
        
        EMatrixTrans = getEMatrixTrans(elemSet[i,:], nodeSet)

        # EMatrix = getEMatrixStiff(elemSet, nodeSet, Y, H, B)
        
        # EMatrixTrans = getEMatrixTrans(elemSet, nodeSet)

        GM[np.ix_(sctrVct, sctrVct)] += EMatrixTrans.T @ EMatrix @ EMatrixTrans
        
    return GM

def getEMatrixStiff(elemSet, nodeSet, Y, H, B):
    # nElem = len(elemSet)
    Ke = np.zeros((6,6))
    
    # for i in range(nElem):
    in_ = elemSet[0]
    jn = elemSet[1]
    E = Y
    A = B * H
    I = B * H**3 / 12  
    L = np.linalg.norm(nodeSet[jn,:] - nodeSet[in_,:])
    L2 = L**2
    
    IdL = 6*I/L  
    IdL2 = 12*I/L2
    I_4 = 4*I
    I_2 = 2*I

    Ke[:,:] = E/L * np.array([[A, -A, 0, 0, 0, 0], 
                                [-A, A, 0, 0, 0, 0],
                                [0, 0, IdL2, -IdL2, IdL, IdL],
                                [0, 0, -IdL2, IdL2, -IdL, -IdL],
                                [0, 0, IdL, -IdL, I_4, I_2],
                                [0, 0, IdL, -IdL, I_2, I_4]])

    return Ke

def getEMatrixTrans(elemSet, nodeSet):
    # nElem = len(elemSet[:,1])
    Te = np.zeros((6,6))
    
    # for i in range(nElem):
    in_ = elemSet[0]
    jn = elemSet[1]
    
    theta = np.arctan2(nodeSet[jn,1]-nodeSet[in_,1], nodeSet[jn,0]-nodeSet[in_,0])
    Ecos = np.cos(theta)
    Esin = np.sin(theta)
    
    Te[:,:] = np.array([[Ecos, 0, Esin, 0, 0, 0],
                            [0, Ecos, 0, Esin, 0, 0],
                            [-Esin, 0, Ecos, 0, 0, 0],
                            [0, -Esin, 0, Ecos, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
                              
    return Te

# getGTStiffness
def getGTStiffness(elemSet, nodeSet, Y, H, B, u, alpha0):

    nDOF = 3
    nNode = len(nodeSet)
    nElem = len(elemSet)
    
    K = csc_matrix((nNode*nDOF, nNode*nDOF))
    f = np.zeros(nNode*nDOF)
    
    for i in range(nElem):
       
        sctr = elemSet[i,:]
        sctrVct = [sctr[0], sctr[1], sctr[0]+nNode, sctr[1]+nNode, sctr[0]+2*nNode, sctr[1]+2*nNode]
        
        Pg = u[sctrVct]
       
        Kg, Fg, alpha = getEMatrixTStiff(elemSet[i,:], nodeSet, Y[i], H[i], B[i], Pg, alpha0[i])
        alpha0[i] = alpha
        K[np.ix_(sctrVct, sctrVct)] += Kg 
        f[sctrVct] += Fg

    return K, f, alpha0

# getEMatrixTStiff 
def getEMatrixTStiff(elemSet, nodeSet, Y, H, B, Pg, alpha0):

    # element properties
    E = Y  
    A = B*H
    I = B*H**3/12
    
    # nodal coordinates
    nodeI = elemSet[0]
    nodeJ = elemSet[1]
    x1 = nodeSet[nodeI,0]
    x2 = nodeSet[nodeJ,0]
    y1 = nodeSet[nodeI,1]
    y2 = nodeSet[nodeJ,1]
    
    # displacements 
    # u1 = Pg[0]; w1 = Pg[1]; theta1 = Pg[2]
    # u2 = Pg[3]; w2 = Pg[4]; theta2 = Pg[5]

    u1 = Pg[0]
    w1 = Pg[2]
    theta1 = Pg[4]
    u2 = Pg[1]
    w2 = Pg[3]
    theta2 = Pg[5]
    
    # geometry
    L0 = np.linalg.norm([x2-x1, y2-y1])  
    Ln = np.linalg.norm([x1+u1-x2-u2, y1+w1-y2-w2]) 
    
    # direction cosines
    co = (x2-x1)/L0 
    so = (y2-y1)/L0
    c = (x2+u2-x1-u1)/Ln  
    s = (y2+w2-y1-w1)/Ln
    
    # incremental angle
    sin_a = co*s - so*c
    cos_a = co*c + so*s
    alpha = atan2(sin_a, cos_a)
    
    # handle cycling of angle  
    if abs(alpha0)-pi/2 > 0:
        if np.sign(alpha) != np.sign(alpha0):
            alpha += np.sign(alpha0)*2*pi

    # local displacements
    u_hat = Ln - L0
    theta1_hat = theta1 - alpha 
    theta2_hat = theta2 - alpha
    
    # local forces
    v215 = 2/15
    
    f1 = u_hat/L0 + (theta1_hat**2 - 0.5*theta1_hat*theta2_hat + theta2_hat**2)/15  
    f2 = v215*theta1_hat - theta2_hat/30
    f3 = v215*theta2_hat - theta1_hat/30
    
    N = E*A*f1
    EAL0f1 = N*L0
    EIL0=E*I/L0
    M1=EAL0f1*f2+EIL0*(4*theta1_hat+2*theta2_hat)
    M2=EAL0f1*f3+EIL0*(2*theta1_hat+4*theta2_hat)
    
    r = np.array([-c, c, -s, s, 0, 0])
    q = np.array([ s, -s, -c, c, 0, 0])
    
    # global forces
    
    
    # Kl(2,2)=E*(AL0*(f2*f2+f1215)+4*IdL0);
    # Kl(2,3)=E*(AL0*(f2*f3-f1/30)+2*IdL0);
    # Kl(3,3)=E*(AL0*(f3*f3+f1215)+4*IdL0);

    IdL0 = I/L0
    # stiffness matrix 
    Kl = np.array([[E*A/L0, E*A*f2, E*A*f3],
                  [E*A*f2, E*(A*L0*(f2**2 + v215*f1) + 4*IdL0), E*(A*L0*(f2*f3 - f1/30) + 2*IdL0)], 
                  [E*A*f3, E*(A*L0*(f2*f3 - f1/30) + 2*IdL0), E*(A*L0*(f3**2 + v215*f1) + 4*IdL0)]])
                  
    B = np.array([[-c, c, -s, s, 0, 0],
                 [-s/Ln, s/Ln, c/Ln, -c/Ln, 1, 0], 
                 [-s/Ln, s/Ln, c/Ln, -c/Ln, 0, 1]])
    
    Fg = B.T @ np.array([N, M1, M2])
                 
    Kg = B.T @ Kl @ B + (np.outer(q,q)*N + (np.outer(r,q) + np.outer(q,r))*0.5*(M1+M2)/Ln)/Ln**2
    
    return Kg, Fg.flatten(), alpha