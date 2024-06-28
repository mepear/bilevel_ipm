import numpy as np
from Net import constraints_net, constraints_adv, cap_cost
from Misc import proj_poly, proj_rplus, maxmin_attack_cost, test, solve_ll
import copy
import math

# TODO TRY DIFFERENT WAYS TO GENERATE RANDOM SAMPLES 
# def sample(G, in_mat, demand, N):
#     m = len(G.edges)
#     edge_cap, edge_cost, cap, cost = cap_cost(G)

#     # Generate random samples uniformly that satisfy capacity constraints
#     ys = np.multiply(np.random.rand(m, N), np.expand_dims(np.array(cap),axis=1))

#     # Constraints in y (LL)
#     A_net, b_net = constraints_net(G, cap, in_mat, demand, pr=0)
#     for j in range(ys.shape[1]):
#         proj_y = proj_poly(A_net, b_net, ys[:,j])
#         ys[:,j] = np.squeeze(proj_y)

#     return ys

# def sample(G, in_mat, demand, N):
#     m = len(G.edges)
#     edge_cap, edge_cost, cap, cost = cap_cost(G)

#     # Generate random samples uniformly that satisfy capacity constraints
#     ys = np.multiply(np.random.rand(m, N), np.expand_dims(np.array(cap),axis=1))

#     return ys

def sample(G, in_mat, demand, N):
    m = len(G.edges)
    edge_cap, edge_cost, cap, cost = cap_cost(G)

    # Generate random samples uniformly that satisfy capacity constraints
    ys = np.multiply(np.random.rand(m, N), np.expand_dims(np.array(cap),axis=1))

    # Constraints in y (LL)
    A_net, b_net = constraints_net(G, cap, in_mat, demand, pr=0)
    ys_new = np.zeros((m,1))
    for j in range(ys.shape[1]):
        proj_y = proj_poly(A_net, b_net, ys[:,j])
        if np.linalg.norm(np.squeeze(proj_y)-ys[:,j]) < 1e-6:
            ys_new = np.hstack((ys_new, ys[:,j]))
    ys_new = np.delete(ys_new,0,1)

    return ys

# Project a vector onto the unit simplex
def sim_proj(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w / np.sum(w)

# f 
def f(x,y): return -np.inner(np.squeeze(x), y)
def gradx_f(x,y): return -y # (m, )
def grady_f(x,y): return -x

# g
def g(x,y,cost): return np.inner((cost + np.squeeze(x)), y) # ()
def gradx_g(x,y): return y # (m, )
def grady_g(x,y,cost): return np.expand_dims(cost) + x

# fbar
def fbar(x, p, ys_list): 
    sum = 0
    for i in range(len(ys_list)): sum += p[i]*ys_list[i]
    return f(x, sum)
def gradx_fbar(x, p, ys_list): 
    sum = 0
    for i in range(len(ys_list)): sum += p[i]*ys_list[i]
    return gradx_f(x, sum) # (m, )
def gradp_fbar(x, ys_list):  
    return -(np.array(ys_list))@x # (num_samp, )

# gbar
def gbar(x, ys_list, p, cost): 
    sum = 0
    for i in range(len(ys_list)): sum += p[i]*g(x, ys_list[i], cost)
    return sum
def gradx_gbar(x, p, ys_list): 
    sum = 0
    for i in range(len(ys_list)): sum += p[i]*gradx_g(x, ys_list[i])
    return sum # (m, )
def gradp_gbar(x, ys_list, cost): 
    g_vec = []
    for j in range(len(ys_list)): g_vec.append(g(x, ys_list[j], cost))
    return np.array(g_vec) # (num_samp, )

def DVFA(G, in_mat, demand, budget, param, pr=0):

    # m: number of edges (size of x,y)
    # k: number of constraints in LL

    edge_cap, edge_cost, cap, cost = cap_cost(G)
    m = len(G.edges)
    # Constraints in y (LL)
    A_net, b_net = constraints_net(G, cap, in_mat, demand, pr=0)
    k = np.shape(A_net)[0]
    # Constraints in x (UL)
    A_adv, b_adv = constraints_adv(G, budget, pr=0)
    k_ul = np.shape(A_adv)[0]
    #
    C = np.eye(m)

    # Step 0: Initialization
    max_iter = param['max_iter']
    step = param['step']
    gamma = param['gamma_dvfa']
    lam = param['lam']
    num_samp = param['num_samp']

    x = np.random.rand(m, 1)
    x = proj_poly(A_adv, b_adv, x)
    p = (1/num_samp)*np.ones(num_samp, )

    # Sample points
    ys = sample(G, in_mat, demand, num_samp)
    ys_list = []
    for j in range(ys.shape[1]):
        ys_list.append(ys[:,j])  # len = num_samp

    # For storing results
    cost_iter = []

    # Generate DVFA attack
    for i in range(max_iter):
        # Step 1: Evaluate Value Function 
        g_vec = []
        for j in range(len(ys_list)): 
            g_vec.append(g(x, ys_list[j], cost)*(-1/lam))
        pstar = sim_proj(np.array(g_vec))

        # Step 2: Calculate gradient of Fgamma 
        sum = 0
        for j in range(len(ys_list)): 
            sum += gradx_g(x, ys_list[j])*pstar[j]
            
        grad_Fx = gradx_fbar(x, p, ys_list) + gamma*(gradx_gbar(x, p, ys_list) - sum)
        grad_Fp = gradp_fbar(x, ys_list) + gamma*(np.expand_dims(gradp_gbar(x, ys_list, cost), axis=1))
        
        # Step 3 
        x = proj_poly(A_adv, b_adv, x - step*np.expand_dims(grad_Fx, axis=1))
        p = sim_proj(p - step*np.squeeze(grad_Fp))

        # Print intermediate results
        if i % 1 == 0:
            edge_flow_tmp, cost_tmp = maxmin_attack_cost(G, x, in_mat, demand)
            cost_iter.append(cost_tmp)
            if i % 5 == 0 and pr == 1:
                print('BVFA: Iter = ', i, 'total cost = ', cost_tmp)
        if math.isinf(cost_tmp):
            break

    # Calculate final total cost
    edge_flow_att, final_cost = maxmin_attack_cost(G, x, in_mat, demand)
    return edge_flow_att, final_cost, 0, 0, cost_iter  




