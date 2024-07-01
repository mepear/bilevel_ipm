import numpy as np
import matplotlib.pyplot as plt
import math
import pickle, json
import argparse
import optuna
from Benchmarks import rand_attack, max_cap_attack, uniform_attack
from Misc import min_cost_flow, comp_var, comp_var2
from Net import generate_graph
from SIGD import SIGD
from PDBO import PDBO
from DVFA import DVFA
from LB import LB
from datetime import datetime
import os, joblib

# Run experiments (multiple runs)
def compute_toll(budget, runs, algs, alg_param, graph_runs, pr=0):


    # INITIALIZE
    final_cost_runs, final_relcost_runs, iter_relcost_runs = {}, {}, {}
    for alg in algs+['base']:
        # final cost across all runs
        final_cost_runs[alg] = []
        # final relative cost across all runs
        final_relcost_runs[alg] = []
        # list of arrays (one per run); each array has the relative cost across the algorithm's iteration
        iter_relcost_runs[alg] = []
    
    run, cnt = 0, 0
    cont = False
    graphs_used = []

    # RUN ALGORITHMS
    while run < runs:
        # Sample graph
        rr, G, m, cap, cost, in_mat, demand = graph_runs[cnt]
        cnt += 1

        print('----------RUN = ', run, '----------------')
        # Before attack
        _, total_cost_base = min_cost_flow(G, cap, cost, in_mat, demand)

        final_cost, iter_cost = {}, {}
        for alg in algs:
            print(f'alg={alg}')
            func = globals()[alg]
            param = alg_param[alg]
            args = (G, in_mat, demand, budget, param, pr, )
            _, final_cost[alg], _, _, iter_cost[alg] = func(*args) 
            if math.isinf(final_cost[alg]):
                cont = True
            if pr == 1:
                print('----------------------------------------')
        
        if cont==True:
            cont = False
            continue

        final_cost_runs['base'].append(total_cost_base)
        print('-----------   RESULTS   ----------------')
        print('Budget = ', budget, ' Run = ', run)
        # print('Cost (before attack) = ', total_cost_base)
        for alg in algs:
            final_cost_runs[alg].append(final_cost[alg])
            final_relcost_runs[alg].append((final_cost[alg] - total_cost_base) / total_cost_base)
            iter_relcost_runs[alg].append((np.array(iter_cost[alg]) - total_cost_base)/total_cost_base)
            print(f'Relative Cost (after {alg} attack) = ', final_relcost_runs[alg])
        print('----------------------------------------')

        graphs_used.append((rr, G, m, cap, cost, in_mat, demand)) 
        run += 1

    # AVERAGE RESULTS ACROSS RUNS
    avg_final_cost, avg_final_relcost, min_final_relcost, max_final_relcost, \
        avg_iter_relcost, min_iter_relcost, max_iter_relcost = {}, {}, {}, {}, {}, {}, {}
    avg_final_cost['base'] = np.round(np.mean(np.array(final_cost_runs['base'])))
    for alg in algs:
        avg_final_cost[alg] = np.round(np.mean(np.array(final_cost_runs[alg])))
        avg_final_relcost[alg], min_final_relcost[alg], max_final_relcost[alg] = comp_var(final_relcost_runs[alg])
        avg_iter_relcost[alg], min_iter_relcost[alg], max_iter_relcost[alg] = comp_var2(np.array(iter_relcost_runs[alg]))


    return avg_final_cost, avg_final_relcost, min_final_relcost, max_final_relcost, avg_iter_relcost, min_iter_relcost, max_iter_relcost, graphs_used


# Run single trial
def toll_set_trial(prob_param, algs, args, graph_param, trial):
        
    # Unpack problem parameters and graphs
    budgets, runs, max_iter = prob_param['budgets'], prob_param['runs'], prob_param['max_iter']
    graph_runs = graph_param['graph_runs']

    # Algorithm parameters (gamma in mgd is reg in pdbo)
    alg_param = {}
    # DVFA
    if 'DVFA' in algs:
        DVFA = {}
        step = trial.suggest_float("step", args.step_l, args.step_u)
        gamma_dvfa = trial.suggest_float("gamma_dvfa", args.gamma_l_dvfa, args.gamma_u_dvfa)
        lam = trial.suggest_float("lambda", args.lambda_l, args.lambda_u)
        num_samp = trial.suggest_int("num_samp", args.num_samp_l, args.num_samp_u)
        DVFA = {'max_iter': max_iter, 'step': step, 'gamma_dvfa': gamma_dvfa, 'lam': lam, 'num_samp': num_samp}
        alg_param['DVFA'] = DVFA

    # SIGD
    if 'SIGD' in algs:
        inn_max_iter = trial.suggest_int("inn_max_iter", args.inn_max_iter_l, args.inn_max_iter_u)
        stepx = trial.suggest_float("stepx", args.stepx_l, args.stepx_u)
        stepy = trial.suggest_float("stepy", args.stepy_l, args.stepy_u)
        stepl = trial.suggest_float("stepl", args.stepl_l, args.stepl_u)
        gamma = trial.suggest_float("gamma", args.gamma_l, args.gamma_u)
        rho = trial.suggest_int("rho", args.rho_l, args.rho_u)
        SIGD = {'max_iter': max_iter, 'inn_max_iter': inn_max_iter, 'stepx': stepx, 'stepy': stepy, 'stepl': stepl,
                'gamma': gamma, 'rho': rho, 'll_solver': args.sigd_ll_solver, 'armijo': args.armijo}
        alg_param['SIGD'] = SIGD
    
    # PDBO
    # TODO
        
    # LB
    if 'LB' in algs:
        t = trial.suggest_float("t", args.t_l, args.t_u)
        step_out = trial.suggest_float("step_out", args.step_out_l, args.step_out_u)
        lb_gamma = trial.suggest_float("step_out", args.lb_gamma_l, args.lb_gamma_u)
        LB = {'max_iter': max_iter, 't': t, 'step_out': step_out, 'gamma': lb_gamma, 'll_solver': args.lb_ll_solver}
        alg_param['LB'] = LB

    # Results across budgets
    avg_final_relcost_bud, min_final_relcost_bud, max_final_relcost_bud, \
        avg_iter_relcost_bud, min_iter_relcost_bud, max_iter_relcost_bud = {}, {}, {}, {}, {}, {}

    for alg in algs:
        avg_final_relcost_bud[alg], min_final_relcost_bud[alg], max_final_relcost_bud[alg], \
            avg_iter_relcost_bud[alg], min_iter_relcost_bud[alg], max_iter_relcost_bud[alg] = [], [], [], [], [], []

    # Run experiements
    for budget in budgets:
        avg_final_cost, avg_final_relcost, min_final_relcost, max_final_relcost, \
        avg_iter_relcost, min_iter_relcost, max_iter_relcost, graphs_used = compute_toll(budget, runs, algs, alg_param, graph_runs)

        print('Budget = ', budget)
        # print('(Average) Cost (before attack) = ', avg_final_cost['base'])
        for alg in algs:
            avg_final_relcost_bud[alg].append(avg_final_relcost[alg])
            min_final_relcost_bud[alg].append(np.array(min_final_relcost[alg]))
            max_final_relcost_bud[alg].append(np.array(max_final_relcost[alg]))
            avg_iter_relcost_bud[alg].append(np.array(avg_iter_relcost[alg]))
            min_iter_relcost_bud[alg].append(np.array(min_iter_relcost[alg]))
            max_iter_relcost_bud[alg].append(np.array(max_iter_relcost[alg]))
            print('(Average) Relative Cost (after {alg} attack) = ', avg_final_relcost[alg])

    curr_res = {'alg_param': alg_param, 'graphs_used': graphs_used,
                'avg_final_relcost_bud': avg_final_relcost_bud, 'min_final_relcost_bud': min_final_relcost_bud, 'max_final_relcost_bud': max_final_relcost_bud, 
                'avg_iter_relcost_bud': avg_iter_relcost_bud, 'min_iter_relcost_bud': min_iter_relcost_bud, 'max_iter_relcost_bud': max_iter_relcost_bud}

    trial.set_user_attr('curr_res', curr_res)
    trial.set_user_attr('trial_num', trial.number)

    avg_list = []
    for alg in algs:
        avg_list.append(avg_final_relcost_bud[alg])
    min = np.min(np.mean(np.array(avg_list), axis=1))

    return min


# Perform hyperparameter optimization 
def hyperopt(args):

    # Problem parameters
    budgets = list(np.arange(args.budgets_l, args.budgets_u, args.budgets_step))
    prob_param = {'budgets': budgets, 'runs': args.runs, 'max_runs': args.max_runs, 'max_iter': args.max_iter}
    # Algorithm selection
    algs = args.algs
    # Graph parameters
    graph_param = {'n': args.n, 'p': args.p, 'cap': (args.cap_l, args.cap_u), 'cost': (args.cost_l, args.cost_u), 'dem':args.dem}
    # Generate many random instances of graphs
    graph_runs = []
    for rr in range(args.max_runs):
        # Create graph
        G, m, cap, cost, in_mat, demand = generate_graph(graph_param['n'], graph_param['p'], graph_param['cap'], 
                                                            graph_param['cost'], graph_param['dem'], 0)
        graph_runs.append((rr, G, m, cap, cost, in_mat, demand))
    graph_param['graph_runs'] = graph_runs
    
    # Create folder for saving parameters and results
    path = 'Results/toll_' + f"{datetime.now().strftime('%m%d%H%M%S%f')}/"
    os.mkdir(path)

    # Save general parameters 
    gen_param = {'prob_param': prob_param, 'graph_param': graph_param, 'algs': algs}
    pickle.dump(gen_param, open(path+"gen_param.p", "wb"))
                 
    # Setup hyperopt study
    study = optuna.create_study(direction='maximize')
    for i in range(args.num_trials):
        # Run single trial
        study.optimize(lambda trial: toll_set_trial(prob_param, algs, args, graph_param, trial), n_trials=1)
        joblib.dump(study, path + "study.pkl")

        # Current trial
        trial_num = study.trials[-1].user_attrs['trial_num']
        trial_path = path + 'Trial_' + str(trial_num) + '/'
        os.mkdir(trial_path)

        # Save trial parameters and results
        curr_res = study.trials[trial_num].user_attrs['curr_res']
        pickle.dump(curr_res, open(trial_path + "curr_res.p", "wb"))

        # Save plots 
        plt = plot(budgets, algs, curr_res['avg_final_relcost_bud'], curr_res['min_final_relcost_bud'], curr_res['max_final_relcost_bud'])
        plt.savefig(trial_path + "relcost_budget.pdf")
        plt = plot_iter(budgets, algs, curr_res['avg_iter_relcost_bud'], curr_res['min_iter_relcost_bud'], curr_res['max_iter_relcost_bud'])
        plt.savefig(trial_path + "relcost_iter.pdf")

        # Best parameters
        best_params = study.best_params
        best_value = study.best_value
        best_trial_num = study.best_trial.number
        best_res = {'best_params': best_params, 'best_value': best_value, 'best_trial_num': best_trial_num}
        json.dump(best_res, open(path + 'best_res.json', 'w'), indent=2)

# Parse input
def parse_input():
    #region Parse parameters
    parser = argparse.ArgumentParser() 
    parser.add_argument('--pc', type=str, default='remote')
    parser.add_argument('--num_trials', type=int, default=2)
    # Problem parameters 
    parser.add_argument('--budgets_l', type=float, default=100)
    parser.add_argument('--budgets_u', type=float, default=105)
    parser.add_argument('--budgets_step', type=float, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--max_runs', type=int, default=1)
    parser.add_argument('--algs', type=str, nargs='+', default=['LB']) 
    # Graph parameters
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--cap_l', type=int, default=100)
    parser.add_argument('--cap_u', type=int, default=1000)
    parser.add_argument('--cost_l', type=int, default=5)
    parser.add_argument('--cost_u', type=int, default=30)
    parser.add_argument('--dem', type=float, default=0.75)
    # Algorithm parameters
    parser.add_argument('--max_iter', type=int, default=5)
    # DVFA
    parser.add_argument('--step_l', type=float, default=1*1e-3)
    parser.add_argument('--step_u', type=float, default=1*1e-3)
    parser.add_argument('--gamma_l_dvfa', type=float, default=1)
    parser.add_argument('--gamma_u_dvfa', type=float, default=1e2)
    parser.add_argument('--lambda_l', type=float, default=1e-3)
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--num_samp_l', type=int, default=1)
    parser.add_argument('--num_samp_u', type=int, default=1)
    # SIGD
    parser.add_argument('--inn_max_iter_l', type=int, default=5)
    parser.add_argument('--inn_max_iter_u', type=int, default=20)
    parser.add_argument('--stepx_l', type=float, default=1*1e-3)
    parser.add_argument('--stepx_u', type=float, default=1*1e-2)
    parser.add_argument('--stepy_l', type=float, default=1*1e-3)
    parser.add_argument('--stepy_u', type=float, default=1*1e-2)
    parser.add_argument('--stepl_l', type=float, default=1*1e-2)
    parser.add_argument('--stepl_u', type=float, default=1*1e-1)
    parser.add_argument('--gamma_l', type=float, default=1*1e-2)
    parser.add_argument('--gamma_u', type=float, default=1*1e-1)
    parser.add_argument('--rho_l', type=int, default=1)
    parser.add_argument('--rho_u', type=int, default=2)
    parser.add_argument('--sigd_ll_solver', type=str, default='exact')
    parser.add_argument('--armijo', type=int, default=0)
    # PDBO

    # LB
    parser.add_argument('--t_l', type=float, default=1e7)
    parser.add_argument('--t_u', type=float, default=1e8)
    parser.add_argument('--step_out_l', type=float, default=1*1e-3)
    parser.add_argument('--step_out_u', type=float, default=2*1e-3)
    parser.add_argument('--lb_gamma_l', type=float, default=1*1e-3)
    parser.add_argument('--lb_gamma_u', type=float, default=1*1e-2)
    parser.add_argument('--lb_ll_solver', type=str, default='exact')

    args = parser.parse_args() 
    
    return args


# PLOT RESULTS
def plot(budgets, algs, avg_final_relcost_bud, min_final_relcost_bud, max_final_relcost_bud):

    fig = plt.figure()
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.tight_layout()

    cols = ['red', 'blue', 'orange', 'green', 'black'][:len(algs)]
    for alg, col in zip(algs,cols):
        plt.plot(budgets, avg_final_relcost_bud[alg], label=alg, color=col, linewidth=3.0, linestyle='-')
        plt.fill_between(budgets, min_final_relcost_bud[alg], max_final_relcost_bud[alg], color=col, alpha=0.1)

    plt.xlabel('Toll budget')
    plt.ylabel('Relative total cost increase')
    plt.legend(loc="upper left")
    # plt.show()

    return plt


def plot_iter(budgets, algs, avg_iter_relcost_bud, min_iter_relcost_bud, max_iter_relcost_bud):

    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.tight_layout()

    for bud in range(len(budgets)):

        fig = plt.figure()

        cols = ['red', 'blue', 'orange', 'green', 'black'][:len(algs)]
        for alg, col in zip(algs,cols):
            iters = list(np.arange((avg_iter_relcost_bud[alg][bud]).shape[0]))
            plt.plot(iters, avg_iter_relcost_bud[alg][bud], label=alg, color=col, linewidth=3.0, linestyle='-')
            plt.fill_between(iters, min_iter_relcost_bud[alg][bud], max_iter_relcost_bud[alg][bud], color=col, alpha=0.1)

        plt.xlabel('Iterations')
        plt.ylabel('Relative total cost increase')
        # plt.legend(loc="upper left")
        plt.legend(loc="best")
        # plt.title('Budget=', bud)

    # plt.show()
        
    return plt


# RUN PROGRAM
if __name__ == "__main__":
    args = parse_input()
    hyperopt(args)