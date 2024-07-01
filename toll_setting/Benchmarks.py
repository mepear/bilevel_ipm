import numpy as np
from Net import constraints_adv, cap_cost, copy_graph
from Misc import min_cost_flow, proj_poly


# ATTACKS
# Random attack
def rand_attack(G, cap, in_mat, demand, budget):
    m = len(G.edges)
    # Constraints in y
    A_adv, b_adv = constraints_adv(G, budget, pr=0)
    # Generate random attack
    y_rnd = np.random.rand(m, 1)
    y_rnd = np.round(proj_poly(A_adv, b_adv, y_rnd), 2)
    # Define new graph
    G_rnd = copy_graph(G)
    for i, (u, v) in enumerate(G_rnd.edges()):
        G_rnd.edges[u, v]['c'] = G_rnd.edges[u, v]['c'] + y_rnd[i, 0]
    # Compute min flow in attacked graph
    edge_cap_rnd, edge_cost_rnd, cap_rnd, cost_rnd = cap_cost(G_rnd)
    edge_flow_rnd, total_cost_rnd = min_cost_flow(G_rnd, cap_rnd, cost_rnd, in_mat, demand)

    return edge_flow_rnd, total_cost_rnd


# Max capacity (add whole toll budget to edge with largest capacity; if more than one add only to one)
def max_cap_attack(G, cap, in_mat, demand, budget):
    m = len(G.edges)
    max = np.max(cap)
    # Define new graph
    G_max = copy_graph(G)
    for i, (u, v) in enumerate(G_max.edges()):
        if G_max.edges[u, v]['cap'] == max:
            G_max.edges[u, v]['c'] = G_max.edges[u, v]['c'] + budget
            break
    # Compute min flow in attacked graph
    edge_cap_max, edge_cost_max, cap_max, cost_max = cap_cost(G_max)
    edge_flow_max, total_cost_max = min_cost_flow(G_max, cap_max, cost_max, in_mat, demand)
    return edge_flow_max, total_cost_max


# Uniform
def uniform_attack(G, cap, in_mat, demand, budget):
    m = len(G.edges)
    budget_edge = np.floor(budget/m)
    # Define new graph
    G_uni = copy_graph(G)
    for i, (u, v) in enumerate(G_uni.edges()):
        G_uni.edges[u, v]['c'] = G_uni.edges[u, v]['c'] + budget_edge

    # Compute min flow in attacked graph
    edge_cap_uni, edge_cost_uni, cap_uni, cost_uni = cap_cost(G_uni)
    edge_flow_uni, total_cost_uni = min_cost_flow(G_uni, cap_uni, cost_uni, in_mat, demand)
    return edge_flow_uni, total_cost_uni

