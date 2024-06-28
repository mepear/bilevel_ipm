import numpy as np
from Net import constraints_net, constraints_adv, cap_cost
from Misc import proj_poly, proj_rplus, maxmin_attack_cost, test, solve_ll
import copy
import math


def comp_grady(y, gamma, A, b, C):

    Abar = active(A, b, y)
    if Abar is None:
        grady_x = (1 / gamma) * (-C)
    else:
        tmp = np.matmul(Abar, np.transpose(Abar))
        tmp1 = -gamma*np.linalg.inv(tmp + 0.001*np.eye(tmp.shape[0], tmp.shape[0]))
        tmp2 = (1/gamma)*np.matmul(Abar, C)
        gradl_x = np.matmul(tmp1, tmp2)
        tmp3 = np.matmul(np.transpose(Abar), gradl_x)
        grady_x = (1/gamma)*(-C - tmp3)

    return grady_x


def active(A, b, y):
    eps = 0.01
    act_rows = []

    for i in range(np.shape(A)[0]):
        if eps > (np.matmul(A[i, :], y) - b[i]) > -eps:
            act_rows.append(i)

    if len(act_rows) == 0:
        Abar = None
    else:
        Abar = np.expand_dims(A[act_rows[0], :], axis=0)
        for j in range(1, len(act_rows)):
            Abar = np.vstack((Abar, A[act_rows[j], :]))

    return Abar


def f(x, y, cost_mgd):
    ff = -(np.expand_dims(cost_mgd, axis=1) + x).T@y
    return ff


def grady_g_pert(x, y, gamma, cost_mgd, q):
    gg = np.expand_dims(cost_mgd, axis=1) + x + gamma * y + q
    return gg


def armijo(x, d, gF, gamma, cost_mgd, q, A_net, b_net, m):
    # Parameters
    max_iter = 30
    iter_inn_armijo = 20
    step_inn_armijo = 1e-2

    s = 1
    sigma = 1
    beta_arm = 0.9

    e0 = 1
    a = s

    # Compute the value of the approximate implicit function at current x
    x1 = copy.deepcopy(x)
    y1 = proj_poly(A_net, b_net, np.expand_dims(np.random.rand(m), axis=1))
    for j in range(iter_inn_armijo):
        y1 = proj_poly(A_net, b_net, y1 - step_inn_armijo * grady_g_pert(x1, y1, gamma, cost_mgd, q))
    f1 = f(x1, y1, cost_mgd)

    for i in range(1, max_iter):
        x2 = copy.deepcopy(x) + a * d
        y2 = proj_poly(A_net, b_net, np.expand_dims(np.random.rand(m), axis=1))
        for j in range(iter_inn_armijo):
            y2 = proj_poly(A_net, b_net, y2 - step_inn_armijo * grady_g_pert(x2, y2, gamma, cost_mgd, q))

        # if f1 - f(x2,y2) < -sigma*a*gF.T@d - e0/i:
        if f1 - f(x2, y2, cost_mgd) < -sigma * a * gF.T @ d - a * e0 / i:
            # if f1 - f(x2,y2) < -sigma*a*gF.T@d:
            a = beta_arm * a
        else:
            return a

    return a


# Maximin attack (with inner min-max problem)
def SIGD(G, in_mat, demand, budget, SIGD_param, pr=0):

    # m: number of edges (size of x,y)
    # k: number of constraints in LL

    edge_cap_sigd, edge_cost_sigd, cap_sigd, cost_sigd = cap_cost(G)
    m = len(G.edges)
    # Constraints in y (LL)
    A_net, b_net = constraints_net(G, cap_sigd, in_mat, demand, pr=0)
    k = np.shape(A_net)[0]
    # Constraints in x (UL)
    A_adv, b_adv = constraints_adv(G, budget, pr=0)
    k_ul = np.shape(A_adv)[0]
    #
    C = np.eye(m)

    q = np.expand_dims(np.random.rand(m), axis=1)
    q = 1e-3 * (q / np.linalg.norm(q))

    max_iter = SIGD_param['max_iter']
    inn_max_iter = SIGD_param['inn_max_iter']
    stepx = SIGD_param['stepx']
    stepy = SIGD_param['stepy']
    stepl = SIGD_param['stepl']
    gamma = SIGD_param['gamma']
    rho = SIGD_param['rho']
    ll_solver = SIGD_param['ll_solver']
    arm = SIGD_param['armijo']

    # Initializations
    x = np.random.rand(m, 1)
    y = np.random.rand(m, 1)
    l = np.random.rand(k, 1)
    x = proj_poly(A_adv, b_adv, x)
    y = proj_poly(A_net, b_net, y)
    l = proj_rplus(l)
    total_cost_iter = []
    gradf_iter = []

    # print('-------- SIGD --------------')
    # Generate minimax attack
    for i in range(max_iter):
        # Solve LL problem (using Aug-)
        if ll_solver=='aug':
            for j in range(inn_max_iter):
                # Gradient computation
                # gradL_y
                gradg_y = np.matmul(C, y + x) + gamma*y
                sum1 = 0
                sum2 = 0
                for r in range(k):
                    tmp = rho*np.matmul(A_net[r, :], y) + l[r]
                    tmp = np.max([tmp, 0])

                    sum1 += tmp*np.transpose(A_net[r, :])

                    ei = np.zeros(k)
                    ei[r] = 1.0
                    tmp = (tmp-l[r])/rho
                    sum2 += tmp*ei

                gradL_y = gradg_y + np.expand_dims(sum1, axis=1)
                gradL_l = np.expand_dims(sum2, axis=1)

                # y step
                y = y - stepy * gradL_y
                # l step
                l = l + stepl * gradL_l
                l = proj_rplus(l)
        elif ll_solver == 'pgd':
            for j in range(inn_max_iter):
                gradg_y = np.matmul(C, np.expand_dims(cost_sigd, axis=1) + x) + gamma*y + q  #Added pertrubation q
                y = proj_poly(A_net, b_net, y - stepy*gradg_y)
        elif ll_solver == 'exact':
            y = solve_ll(x, A_net, b_net, gamma, C, cost_sigd)
        else:
            print('Error')

        #Solve UL problem
        gradf_x = -np.matmul(C, y)
        gradf_y = -np.matmul(C, np.expand_dims(cost_sigd, axis=1) + x)
        grady_x = comp_grady(y, gamma, A_net, b_net, C)
        gradF = gradf_x + np.matmul(grady_x, gradf_y)
        if arm == 1:
            stepx = armijo(x, y, gradF, gamma, cost_sigd, q, A_net, b_net, m)

        x = x - stepx * gradF
        x = proj_poly(A_adv, b_adv, x)

        # Print intermediate results
        if i % 1 == 0:
            edge_flow_tmp, total_cost_tmp = maxmin_attack_cost(G, x, in_mat, demand)
            total_cost_iter.append(total_cost_tmp)
            if i % 5 == 0 and pr == 1:
                print('SIGD: Iter = ', i, 'total cost = ', total_cost_tmp)
                # print('Attack flow ', np.round(y.T,1))
        if math.isinf(total_cost_tmp):
            break

    # Calculate final total cost
    edge_flow_att, total_cost_att = maxmin_attack_cost(G, x, in_mat, demand)
    return edge_flow_att, total_cost_att, 0, 0, total_cost_iter 

