import numpy as np
from scipy.optimize import minimize

def analyze_prbm(X0, Y0, Con, BC, F, gamma, kth, kex):
    NNODE = len(X0)
    NELEM = len(Con)
    theta = np.zeros(NELEM)
    L = np.zeros(NELEM)

    if isinstance(kth, (int, float)):
        kth = np.ones(NELEM) * kth

    if isinstance(kex, (int, float)):
        kex = np.ones(NELEM) * kex

    for i in range(NELEM):
        delta_x = X0[Con[i, 1]-1] - X0[Con[i, 0]-1]
        delta_y = Y0[Con[i, 1]-1] - Y0[Con[i, 0]-1]
        L[i] = np.linalg.norm([delta_x, delta_y])
        nx = delta_x / L[i]
        ny = delta_y / L[i]
        theta[i] = np.arctan(np.abs(ny) / np.abs(nx))
        if ny >= 0 and nx < 0:
            theta[i] = np.pi - theta[i]
        if ny < 0 and nx < 0:
            theta[i] = np.pi + theta[i]
        if ny < 0 and nx >= 0:
            theta[i] = 2 * np.pi - theta[i]

    # Design variables
    # [X coordinate, Y coordinate, rigid body rotation alphar, alpha1, delta l, alpha2]
    alphar0 = np.zeros(NELEM)
    alpha10 = np.zeros(NELEM)
    alpha20 = np.zeros(NELEM)
    deltal0 = np.zeros(NELEM)

    Var0 = np.concatenate((X0, Y0, alphar0, alpha10, alpha20, deltal0))
    arguments = (X0, Y0, L, F, NNODE, NELEM, gamma, kth, kex)
    alpharub = alphar0 + np.pi / 2
    alpharlb = alphar0 - np.pi / 2
    alpha1ub = alpha10 + np.pi / 2
    alpha1lb = alpha10 - np.pi / 2
    alpha2ub = alpha20 + np.pi / 2
    alpha2lb = alpha20 - np.pi / 2
    deltalub = deltal0 + np.min(L) / 2
    deltallb = deltal0 - np.min(L) / 2

    Lb = np.concatenate((X0 - 100, Y0 - 100, alpharlb, alpha1lb, alpha2lb, deltallb))
    Ub = np.concatenate((X0 + 100, Y0 + 100, alpharub, alpha1ub, alpha2ub, deltalub))
    bounds = np.zeros((len(Var0), 2))
    for i in range(len(Var0)):
        bounds[i, :] = np.array([Lb[i], Ub[i]])

    constraints={
        'type': 'eq', 
        'fun': constraint,
        'args': (X0, Y0, L, theta, Con, BC, NNODE, NELEM, gamma)
    },

    options = {
        'disp': True,
        # 'maxfun': 1e10, 
        # 'ftol': 1e-10,
        'maxiter': 300
    }

    res = minimize(
        objective,
        Var0,
        method='trust-constr',
        options=options,
        args=arguments, 
        bounds=bounds, 
        constraints=constraints,
        tol=1e-10
    )
    Var = res.x

    X = Var[:NNODE]
    Y = Var[NNODE:2 * NNODE]
    alphar = Var[2 * NNODE:2 * NNODE + NELEM]
    alpha1 = Var[2 * NNODE + NELEM:2 * NNODE + 2 * NELEM]
    alpha2 = Var[2 * NNODE + 2 * NELEM:2 * NNODE + 3 * NELEM]
    deltaL = Var[2 * NNODE + 3 * NELEM:2 * NNODE + 4 * NELEM]
    
    langrage_multipliers = res.v[0]
    Ri = np.zeros(3 * NNODE)
    for i in range(len(BC)):
        Ri[(BC[i, 0]-1) + NNODE * (BC[i, 1]-1)] = -langrage_multipliers[i]

    for i in range(len(F)):
        if F[i, 2] != 0:
            Ri[(F[i, 0]-1) + NNODE * F[i, 1]] = F[i, 2]

    print(Ri)
    return X, Y, alphar, alpha1, alpha2, Ri


def constraint(Var, X0, Y0, L, theta, Con, BC, NNODE, NELEM, gamma):
    Lr = (1 - gamma) * L / 2
    X = Var[:NNODE]
    Y = Var[NNODE:2 * NNODE]
    alphar = Var[2 * NNODE:2 * NNODE + NELEM]
    alpha1 = Var[2 * NNODE + NELEM:2 * NNODE + 2 * NELEM]
    alpha2 = Var[2 * NNODE + 2 * NELEM:2 * NNODE + 3 * NELEM]
    deltaL = Var[2 * NNODE + 3 * NELEM:2 * NNODE + 4 * NELEM]

    elemN = np.zeros((NNODE, NELEM))
    cbound = np.zeros(len(BC))
    ccon=np.zeros(NELEM*2)

    # kinematic constraint
    for i in range(NELEM):
        j = Con[i, 0] - 1
        k = Con[i, 1] - 1
        ccon[i * 2] = X[j] + Lr[i] * np.cos(theta[i] + alphar[i]) + (gamma * L[i] + deltaL[i]) * np.cos(theta[i] + alphar[i] + alpha1[i]) + Lr[i] * np.cos(theta[i] + alphar[i] + alpha1[i] + alpha2[i]) - X[k]
        ccon[i * 2 + 1] = Y[j] + Lr[i] * np.sin(theta[i] + alphar[i]) + (gamma * L[i] + deltaL[i]) * np.sin(theta[i] + alphar[i] + alpha1[i]) + Lr[i] * np.sin(theta[i] + alphar[i] + alpha1[i] + alpha2[i]) - Y[k]

        elemN[j, i] = alphar[i]
        elemN[k, i] = alphar[i] + alpha1[i] + alpha2[i]
    
    # Boundary condition constraints
    for i in range(len(BC)):
        if BC[i, 1] == 1:
            cbound[i] = X[BC[i, 0]-1] - X0[BC[i, 0]-1] - BC[i, 2]
        elif BC[i, 1] == 2:
            cbound[i] = Y[BC[i, 0]-1] - Y0[BC[i, 0]-1] - BC[i, 2]
        elif BC[i, 1] == 3:
            nd = BC[i, 0] - 1
            for j in range(NELEM):
                if Con[j, 0]-1 == nd or Con[j, 1]-1 == nd:
                    cbound[i] = elemN[nd, j] - BC[i, 2]

    # Ensuring no relative rotations at the interface between two elements
    crot2 = []
    for i in range(NNODE):
        niter = 0
        for j in range(NELEM):
            if Con[j, 0]-1 == i or Con[j, 1]-1 == i:
                if niter == 0:
                    crow = elemN[i, j]
                    niter += 1
                else:
                    crot2.append(elemN[i, j] - crow)
                    niter += 1

    ceq = np.concatenate((cbound, ccon, crot2))

    return ceq

def objective(Var, X0, Y0, L, F, NNODE, NELEM, gamma, kth, kex):
    Lr = (1 - gamma) * L / 2
    X = Var[:NNODE]
    Y = Var[NNODE:2 * NNODE]
    alphar = Var[2 * NNODE:2 * NNODE + NELEM]
    alpha1 = Var[2 * NNODE + NELEM:2 * NNODE + 2 * NELEM]
    alpha2 = Var[2 * NNODE + 2 * NELEM:2 * NNODE + 3 * NELEM]
    deltaL = Var[2 * NNODE + 3 * NELEM:2 * NNODE + 4 * NELEM]

    v = 0

    for i in range(NELEM):
        v += 0.5 * kth[i] * alpha1[i] ** 2 + 0.5 * kth[i] * alpha2[i] ** 2 + 0.5 * kex[i] * deltaL[i] ** 2

    for i in range(len(F)):
        if F[i, 1] == 1:
            v -= F[i, 2] * (X[F[i, 0]-1] - X0[F[i, 0]-1])
        elif F[i, 1] == 2:
            v -= F[i, 2] * (Y[F[i, 0]-1] - Y0[F[i, 0]-1])

    return v


if __name__ == "__main__":
    elemSet = np.array([[1, 2],
                        [2, 3],
                        [2, 4]])

    nodeSet = np.array([[0, 0],  # 1
                        [10, 20],  # 2
                        [40, 20],  # 3
                        [40, 0]])  # 4

    B = np.ones(len(elemSet)) * 5  # out-of-plane thickness
    H = np.ones(len(elemSet)) * 5  # in-plane thickness
    Y = np.ones(len(elemSet)) * 200e2  # Young's Modulus

    # Applied forces / torque
    # f = [a, b, c]
    # a is node number
    # b is the degree of freedom of the corresponding node: x=1, y=2, theta=3
    # c is the value 
    f = np.array([[4, 2, 0]])


    # Boundary condition
    # BC = [a, b, c]
    # a is node number
    # b is the degree of freedom of the corresponding node: x=1, y=2, theta=3
    # c is the value of displacement
    bc = np.array([[1, 1, 0],
                   [1, 2, 0],
                   [1, 3, 0],
                   [3, 1, 0],
                   [3, 3, 0],
                   [4, 1, 0],
                   [4, 3, 0],
                   [4, 2, 5]])  # applied boundary conditions

    # Define PRBM parameters
    gamma = 0.85

    nElem = len(elemSet)
    nNode = len(nodeSet)
    X0 = nodeSet[:, 0]
    Y0 = nodeSet[:, 1]
    kth = np.zeros(nElem)
    kex = np.zeros(nElem)
    L = np.zeros(nElem)
    theta = np.zeros(nElem)

    for i in range(nElem):
        delta_x = X0[elemSet[i, 1]-1] - X0[elemSet[i, 0]-1]
        delta_y = Y0[elemSet[i, 1]-1] - Y0[elemSet[i, 0]-1]
        L[i] = np.linalg.norm([delta_x, delta_y])
        nx = delta_x / L[i]
        ny = delta_y / L[i]
        theta[i] = np.arctan(np.abs(ny) / np.abs(nx))
        if ny >= 0 and nx < 0:
            theta[i] = np.pi - theta[i]
        if ny < 0 and nx < 0:
            theta[i] = np.pi + theta[i]
        if ny < 0 and nx >= 0:
            theta[i] = 2 * np.pi - theta[i]

        kth[i] = 2 * 2.67 * gamma * Y[i] * B[i] * H[i] ** 3 / 12 / L[i]
        kex[i] = Y[i] * B[i] * H[i] / L[i]
    X, Y, alphar, alpha1, alpha2, Ri = analyze_prbm(nodeSet[:, 0], nodeSet[:, 1], elemSet, bc, f, gamma, kth, kex)

    print("Ri:", Ri)

    # Plot the result
    import matplotlib.pyplot as plt

    fig = plt.figure()

    for i in range(nElem):
        Xplot0 = [X0[elemSet[i, 0]-1], X0[elemSet[i, 1]-1]]
        Yplot0 = [Y0[elemSet[i, 0]-1], Y0[elemSet[i, 1]-1]]
        plt.plot(Xplot0, Yplot0, 'b--')

        Xplot = [X[elemSet[i, 0]-1],
                 X[elemSet[i, 0]-1] + L[i] * (1 - gamma) / 2 * np.cos(theta[i] + alphar[i]),
                 X[elemSet[i, 0]-1] + L[i] * (1 - gamma) / 2 * np.cos(theta[i] + alphar[i]) + L[i] * gamma * np.cos(theta[i] + alphar[i] + alpha1[i]),
                 X[elemSet[i, 1]-1]]
        Yplot = [Y[elemSet[i, 0]-1],
                 Y[elemSet[i, 0]-1] + L[i] * (1 - gamma) / 2 * np.sin(theta[i] + alphar[i]),
                 Y[elemSet[i, 0]-1] + L[i] * (1 - gamma) / 2 * np.sin(theta[i] + alphar[i]) + L[i] * gamma * np.sin(theta[i] + alphar[i] + alpha1[i]),
                 Y[elemSet[i, 1]-1]]
        plt.plot(Xplot, Yplot, 'r')

    plt.axis('equal')
    plt.show()