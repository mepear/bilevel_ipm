#!/bin/bash -l    
#SBATCH --time=24:00:00
#SBATCH -p agsmall                                            


conda activate toll

python3 Toll_setting.py \
--num_trials 400 \
--budgets_l 100 --budgets_u 150 --budgets_step 10 \
--runs 5 --max_runs 100 \
--algs 'DVFA' 'SIGD' 'LB' \
--n 14 --p 1 --cap_l 100 --cap_u 1000 --cost_l 5 --cost_u 30 --dem 0.9 \
--max_iter 50 \
--stepx_l 1e-3 --stepx_u 1e-1 --stepy_l 1e-3 --stepy_u 1e-1 --stepl_l 1e-3 --stepl_u 1e-1 \
--gamma_l 1e-6 --gamma_u 1e-1 --sigd_ll_solver 'exact' --armijo 0 \
--t_l 1e4 --t_u 1e9 --step_out_l 1e-3 --step_out_u 1e-1 --lb_gamma_l 1e-3 --lb_gamma_u 1e-1 --lb_ll_solver 'exact' \
--step_l 1e-3 --step_u 1e-1 --gamma_l_dvfa 1 --gamma_u_dvfa 1e7 --lambda_l 1e-5 --lambda_u 1e-2 --num_samp_l 100 --num_samp_u 500

# aglarge
# agsmall
