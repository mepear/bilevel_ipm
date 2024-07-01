import numpy as np
from Net import constraints_net, constraints_adv, cap_cost
from Misc import proj_poly, proj_rplus, maxmin_attack_cost, test, solve_ll
import copy


# Define constraints matrices for the network player (y player - LL) in the LB reformulation
# In the LB reformulation we do not have the inequality constraints
# Consider as the 0: source node and (n-1): sink node
def constraints_net_lb(G, cap, in_mat, demand, pr=0):
    n = len(G.nodes)
    m = len(G.edges)

    # Conservation of flow
    A2tmp = in_mat[1:(n - 1), :]
    A2 = np.vstack((A2tmp, (-1) * A2tmp))
    b2tmp = np.zeros((n - 2, 1))
    b2 = np.vstack((b2tmp, b2tmp))
    if pr == 1:
        print('A2.shape', A2.shape)
        print('b2.shape', b2.shape)

    # Fix amount at source
    A3tmp = np.expand_dims(in_mat[0, :], 0)
    A3 = np.vstack(((-1) * A3tmp, A3tmp))
    b3tmp = np.array([demand])
    b3 = np.vstack((b3tmp, (-1) * b3tmp))
    if pr == 1:
        print('A3.shape', A3.shape)
        print('b3.shape', b3.shape)

    # Fix amount at sink
    A4tmp = np.expand_dims(in_mat[n - 1, :], 0)
    A4 = np.vstack((A4tmp, (-1) * A4tmp))
    b4tmp = np.array([demand])
    b4 = np.vstack((b4tmp, (-1) * b4tmp))
    if pr == 1:
        print('A4.shape', A4.shape)
        print('b4.shape', b4.shape)

    # Final constraints matrix
    A = np.vstack((A2, A3, A4))
    b = np.vstack((b2, b3, b4))
    if pr == 1:
        print('A.shape', A.shape)
        print('b.shape', b.shape)

    return A, b


def constraints_ineq_lb(G, cap, in_mat, demand, pr=0):
    m = len(G.edges)

    # # 0<=x<=p
    # A1 = np.vstack((np.eye(m), (-1) * np.eye(m)))
    # b1 = np.vstack((np.zeros((m, 1)), np.expand_dims(cap, 1)))
    # if pr == 1:
    #     print('A1.shape', A1.shape)
    #     print('b1.shape', b1.shape)

    # 0x<=p
    A10 = np.eye(m)
    b10 = np.zeros((m, 1))
    if pr == 1:
        print('A10.shape', A10.shape)
        print('b10.shape', b10.shape)

    # 0<=x
    A12 = (-1) * np.eye(m)
    b12 = np.expand_dims(cap, 1)
    if pr == 1:
        print('A12.shape', A12.shape)
        print('b12.shape', b12.shape)

    return A10, b10, A12, b12


# h constraints
def hh(x, y, A1, b1, A2, b2, i, ind):
    # (1,)
    if i == 0:
        tmp = (A1 @ y)[ind] - b1[ind]
    elif i == 1:
        tmp = (A2 @ y)[ind] - b2[ind]
    return tmp


# Compute grad of y*(x)
def comp_grady(y, gamma, A, b, A0, b0, A1, b1, C, t):

    hessyy_phi = gamma*np.eye(y.shape[0]) + (1/t)*np.diag(np.divide(np.ones((y.shape[0],)), (y-b0)))
    hessxy_phi = np.eye(y.shape[0])

    tmp = np.linalg.inv(hessyy_phi + 0.001*np.eye(y.shape[0], y.shape[0]))
    tmp1 = (-1)*np.linalg.inv(A@tmp@A.T + 0.001*np.eye(A.shape[0], A.shape[0]))
    tmp2 = A@tmp@hessxy_phi
    gradl_x = np.matmul(tmp1, tmp2)
    tmp3 = A.T@gradl_x
    grady_x = tmp@(-hessxy_phi - tmp3)

    return grady_x


# LOG-BARRIER ALGORITHM
def LB(G, in_mat, demand, budget, LB_param, pr=0):

    # m: number of edges (size of x,y)
    # k: number of constraints in LL

    edge_cap_lb, edge_cost_lb, cap_lb, cost_lb = cap_cost(G)
    m = len(G.edges)
    # Constraints in y (LL) - all constraints
    A_net, b_net = constraints_net(G, cap_lb, in_mat, demand, pr=0)
    # Constraints in y (LL) - w/o the inequality constraints
    A_net_lb, b_net_lb = constraints_net_lb(G, cap_lb, in_mat, demand, pr=0)
    k = np.shape(A_net)[0]
    # Constraints in y (LL) - the inequality constraints added to the objective using barrier function
    A_in0, b_in0, A_in1, b_in1 = constraints_ineq_lb(G, cap_lb, in_mat, demand, pr=0)
    # Constraints in x (UL)
    A_adv, b_adv = constraints_adv(G, budget, pr=0)
    k_ul = np.shape(A_adv)[0]
    #
    C = np.eye(m)

    max_iter = LB_param['max_iter']
    step_out = LB_param['step_out']
    t = LB_param['t']
    gamma = LB_param['gamma']
    ll_solver = LB_param['ll_solver']

    # Initializations
    x = np.random.rand(m, 1)
    # y = np.random.rand(m, 1)
    x = proj_poly(A_adv, b_adv, x)
    # y = proj_poly(A_net, b_net, y)
    total_cost_iter = []
    gradf_iter = []

    for i in range(max_iter):
        if ll_solver == 'gd':
            pass
        elif ll_solver == 'exact':
            y = solve_ll(x, A_net, b_net, gamma, C, cost_lb)

        #Solve UL problem
        gradf_x = -np.matmul(C, y)
        gradf_y = -np.matmul(C, np.expand_dims(cost_lb, axis=1) + x)

        grady_x = comp_grady(y, gamma, A_net, b_net, A_in0, b_in0, A_in1, b_in1, C, t)
        gradF = gradf_x + grady_x@gradf_y

        x = x - step_out * gradF
        x = proj_poly(A_adv, b_adv, x)

        # Print intermediate results
        if i % 1 == 0:
            edge_flow_tmp, total_cost_tmp = maxmin_attack_cost(G, x, in_mat, demand)
            total_cost_iter.append(total_cost_tmp)
            if i % 5 == 0 and pr == 1:
                print('LB: Iter = ', i, 'total cost = ', total_cost_tmp)
                # print('Attack flow ', np.round(y.T,1))


    # Calculate final total cost
    edge_flow_att, total_cost_att = maxmin_attack_cost(G, x, in_mat, demand)
    return edge_flow_att, total_cost_att, 0, 0, total_cost_iter 