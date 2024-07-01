#!/bin/bash -l    


python3 Toll_setting.py \
--num_trials 100 \
--budgets_l 100 --budgets_u 200 --budgets_step 5 \
--runs 1 --max_runs 100 \
--algs 'DVFA' 'SIGD' 'LB' \
--n 12 --p 0.8 --cap_l 100 --cap_u 1000 --cost_l 5 --cost_u 30 --dem 0.75 \
--max_iter 10 \
--stepx_l 1e-3 --stepx_u 1e-1 --stepy_l 1e-3 --stepy_u 1e-1 --stepl_l 1e-3 --stepl_u 1e-1 \
--gamma_l 1e-6 --gamma_u 1e-1 --sigd_ll_solver 'exact' --armijo 0 \
--t_l 1e4 --t_u 1e9 --step_out_l 1e-3 --step_out_u 1e-1 --lb_gamma_l 1e-3 --lb_gamma_u 1e-1 --lb_ll_solver 'exact' \
--step_l 1e-3 --step_u 1e-1 --gamma_l_dvfa 1 --gamma_u_dvfa 100 --lambda_l 1e-3 --lambda_u 1 --num_samp_l 20 --num_samp_u 100 
