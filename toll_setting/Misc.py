import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers
from Net import constraints_net, copy_graph, cap_cost


def comp_var(cost):
    cost = np.array(cost)
    min = np.min(cost)
    max = np.max(cost)
    avg = np.mean(cost)
    return avg, min, max

def comp_var2(cost):
    cost = np.array(cost)
    min = np.min(cost, axis=0)
    max = np.max(cost, axis=0)
    avg = np.mean(cost, axis=0)
    return avg, min, max

# Project on a polyhedron
def proj_poly(A, b, x):

    solvers.options['feastol'] = 1e-5
    # solvers.options['reltol'] = 1e-14
    # Supress messages
    solvers.options['show_progress'] = False

    # Setup problem
    m = x.shape[0]
    A = matrix(A.astype(float), tc='d')
    b = matrix(b.astype(float), tc='d')
    P = matrix(np.eye(m).astype(float), tc='d')
    q = matrix((-x).astype(float), tc='d')

    # Solve problem
    sol = solvers.qp(P, q, A, b)
    argmin = np.array(sol['x'])

    return argmin


# Projection on nonegative orthant
def proj_rplus(x):
    for i in range(np.shape(x)[0]):
        if x[i] < 0:
            x[i] = 0
    return x


# Calculate min cost flow
def min_cost_flow(G, cap, cost, in_mat, demand):
    # Form constraints
    A, b = constraints_net(G, cap, in_mat, demand, pr=0)

    # Define the CVXPY problem
    m = len(G.edges)
    x = cp.Variable((m, 1))
    obj = (cost.T)@x
    cons = A @ x <= b
    prob = cp.Problem(cp.Minimize(obj), [cons])

    # Solve the CVXPY problem
    prob.solve()

    # Calculate total cost and flow
    if x.value is not None:
        edge_flow = []
        total_cost = 0
        for i, (u, v) in enumerate(G.edges()):
            edge_flow.append([(u, v), round(x.value[i, 0], 2)])
            total_cost += cost[i] * x.value[i, 0]
        total_cost = round(total_cost, 2)
    else:
        print('Demand cannot be met')
        edge_flow = []
        total_cost = float('inf')

    return edge_flow, total_cost


# Implement and compute cost after maxmin attack
def maxmin_attack_cost(G, x, in_mat, demand):  
    # Define new graph
    G_att = copy_graph(G)
    for i, (u, v) in enumerate(G_att.edges()):
        G_att.edges[u, v]['c'] = G_att.edges[u, v]['c'] + x[i, 0]

    # Compute min flow in attacked graph
    edge_cap_att, edge_cost_att, cap_att, cost_att = cap_cost(G_att)
    edge_flow_att, total_cost_att = min_cost_flow(G_att, cap_att, cost_att, in_mat, demand)

    return edge_flow_att, total_cost_att


def test(y, cap, cost, bud):
    for i in range(y.shape[0]):
        if y[i] < 0:
            print('yi negative')
        elif y[i] > cap[i]:
            print('yi capacity violation')
    if np.sum(y) - bud > 1e-6:
        print('budget violation', np.sum(y))


# Find min of the constrained problem
def solve_ll(x, A, b, gamma, C, cost):
    solvers.options['feastol'] = 1e-12
    # solvers.options['reltol'] = 1e-8
    # Supress messages
    solvers.options['show_progress'] = False

    # Setup problem
    m = A.shape[1]
    tmp = (gamma/2)*np.eye(m)
    tmp1 = np.matmul(C, np.expand_dims(cost, axis=1) + x)
    Ac = matrix(A.astype(float), tc='d')
    bc = matrix(b.astype(float), tc='d')
    P = matrix(tmp.astype(float), tc='d')
    q = matrix(tmp1.astype(float), tc='d')

    sol = solvers.qp(P, q, Ac, bc)
    argmin = np.array(sol['x'])

    return argmin