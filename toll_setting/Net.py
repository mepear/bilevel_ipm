import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


# Incidence matrix
def incidence(G):
    n = len(G.nodes)
    m = len(G.edges)
    in_mat = np.zeros((n, m))
    for i, u in enumerate(G.nodes()):
        for j, (v, w) in enumerate(G.edges()):
            if u == v:
                in_mat[i, j] = -1
            elif u == w:
                in_mat[i, j] = 1
    return in_mat


# Returns arrays with the cost and capacity of the edges
def cap_cost(G):
    cap = []
    cost = []
    edge_cap = []
    edge_cost = []
    for i, (u, v) in enumerate(G.edges()):
        # Capacity
        cc1 = G.edges[u, v]['cap']
        cap.append(cc1)
        edge_cap.append([(u, v), cc1])
        # Cost coefficient
        cc2 = G.edges[u, v]['c']
        cost.append(cc2)
        edge_cost.append([(u, v), cc2])

    cap = np.array(cap)
    cost = np.array(cost)
    return edge_cap, edge_cost, cap, cost


# Compute capacity of source
def src_cap(G):
    cap = 0
    for i, (u, v) in enumerate(G.edges()):
        if u == 0:
            cap += G.edges[u, v]['cap']
    return cap


# Compute capacity of sink
def sink_cap(G):
    n = len(G.nodes)
    cap = 0
    for i, (u, v) in enumerate(G.edges()):
        if v == (n - 1):
            cap += G.edges[u, v]['cap']
    return cap


# Print parameters
def print_param(G, edge_cap, edge_cost, in_mat, demand):
    print('Nodes : ', G.nodes)
    print('Edges : ', G.edges)
    print('Num nodes : ', len(G.nodes))
    print('Num edges : ', len(G.edges))
    print('Density : ', len(G.edges) / (len(G.nodes) * (len(G.nodes) - 1)))  # #edges in complete directed graph n*(n-1)
    print('Capacity : ', edge_cap)
    print('Source Capacity : ', src_cap(G))
    print('Sink Capacity : ', sink_cap(G))
    print('Cost coeff. : ', edge_cost)
    print('Demand : ', demand)
    print('Incidence matrix : ')
    print(in_mat)


# Draw graph
def draw_graph(G, graph=0, lab=1):
    if graph == 0:
        pos = nx.spring_layout(G)
    else:
        pos = nx.planar_layout(G)
    nx.draw(G, pos=pos, with_labels=True, font_weight='bold', width=4, node_size=450)
    if lab == 1:
        nx.draw_networkx_edge_labels(G, pos=pos, font_size=14)


# Create a copy of a given directed graph
def copy_graph(G):
    G_copy = nx.DiGraph()
    G_copy.add_nodes_from(G)
    G_copy.add_edges_from(G.edges)
    for i, (u, v) in enumerate(G_copy.edges()):
        G_copy.edges[u, v]['cap'] = G.edges[u, v]['cap']
        G_copy.edges[u, v]['c'] = G.edges[u, v]['c']
    return G_copy


# Define constraints matrices for the network player (y player - LL)
# Consider as the 0: source node and (n-1): sink node
def constraints_net(G, cap, in_mat, demand, pr=0):
    n = len(G.nodes)
    m = len(G.edges)
    # 0<=x<=p
    A1 = np.vstack(((-1) * np.eye(m), np.eye(m)))
    b1 = np.vstack((np.zeros((m, 1)), np.expand_dims(cap, 1)))
    if pr == 1:
        print('A1.shape', A1.shape)
        print('b1.shape', b1.shape)

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
    A = np.vstack((A1, A2, A3, A4))
    b = np.vstack((b1, b2, b3, b4))
    if pr == 1:
        print('A.shape', A.shape)
        print('b.shape', b.shape)

    return A, b


# Define constraints matrices for the toll company (x player - UL)
def constraints_adv(G, budget, pr=0):
    m = len(G.edges)

    # 0<=x
    A1 = (-1) * np.eye(m)
    b1 = np.zeros((m, 1))
    if pr == 1:
        print('A1.shape', A1.shape)
        print('b1.shape', b1.shape)

    # Budget constraints
    A2 = np.vstack((np.ones((1, m)), (-1) * np.ones((1, m))))
    b2tmp = np.array([budget])
    b2 = np.vstack((b2tmp, (-1) * b2tmp))
    if pr == 1:
        print('A2.shape', A2.shape)
        print('b2.shape', b2.shape)

    # Final constraints matrix
    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))
    if pr == 1:
        print('A.shape', A.shape)
        print('b.shape', b.shape)

    return A, b


# Generate graph
# Parameters : nodes, edges, density, capacity, cost, demand, budget
def generate_graph(n, p, cap, cost, dem, pr=0):

    # Create graph (DiGraph)
    G = nx.gnp_random_graph(n, p, directed=True)
    m = len(G.edges)

    # Define capacity and cost coefficients
    t_cap, t_cost = 0, 0
    for (u, v) in G.edges():
        # Capacity
        tmp_cap = round(random.uniform(cap[0],cap[1]), 2)
        t_cap += tmp_cap
        G.edges[u, v]['cap'] = tmp_cap
        # Cost coefficient
        tmp_cost = round(random.uniform(cost[0],cost[1]), 2)
        t_cost += tmp_cost
        G.edges[u, v]['c'] = tmp_cost

    # Demand
    demand = dem*sink_cap(G)  

    # Incidence matrix
    in_mat = incidence(G)
    edge_cap, edge_cost, cap, cost = cap_cost(G)
    # Draw graph and print parameters
    if pr == 1:
        draw_graph(G, 0, 0)
        plt.savefig("graph_main.pdf")
        print_param(G, edge_cap, edge_cost, in_mat, demand)
        print('Total capacity : ', np.sum(cap))


    return G, m, cap, cost, in_mat, demand






