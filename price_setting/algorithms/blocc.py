# BLOCC.py

import numpy as np
import cvxpy as cp
import time

class BLOCC:
    def __init__(self, problem, hparams):
        self.problem = problem
        blocc_params = hparams.get('blocc', {})

        self.alpha_T = blocc_params.get('alpha_T', 1e-4)
        self.gamma = blocc_params.get('gamma', 0.1)
        self.alpha_g_xy = blocc_params.get('alpha_g_xy', 0.1)
        self.alpha_F_xy = blocc_params.get('alpha_F_xy', 0.1)
        self.beta_g_xy = blocc_params.get('beta_g_xy', 1e-6)
        self.beta_F_xy = blocc_params.get('beta_F_xy', 1e-6)
        self.epsilon_T = blocc_params.get('epsilon_T', 1e-6)
        self.epsilon_inner_xy_g = blocc_params.get('epsilon_inner_xy_g', 1e-6)
        self.epsilon_outer_xy_g = blocc_params.get('epsilon_outer_xy_g', 1e-6)
        self.epsilon_inner_xy_F = blocc_params.get('epsilon_inner_xy_F', 1e-6)
        self.epsilon_outer_xy_F = blocc_params.get('epsilon_outer_xy_F', 1e-6)
        self.maxmin_g_outer_max_iters = blocc_params.get('maxmin_g_outer_max_iters', 10000)
        self.maxmin_F_outer_max_iters = blocc_params.get('maxmin_F_outer_max_iters', 10000)
        self.maxmin_g_inner_max_iters = blocc_params.get('maxmin_g_inner_max_iters', 10000)
        self.maxmin_F_inner_max_iters = blocc_params.get('maxmin_F_inner_max_iters', 10000)
        self.main_max_iters = blocc_params.get('main_max_iters', 10000)
    
    def constraint_vector(self, T, x, y):
        constraints = []
    
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(T, x, y, i))
    
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(T, x, y, i))
    
        constraint_vector = np.array(constraints)
        return constraint_vector

    def compute_inner_product_xy(self, T, x, y, mu_vector):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_xy = np.zeros((total_constraints, 2 * n))
        
        for i in range(num_constraints_h1):
            grad_x, grad_y = self.problem.gradient_h_1_xy(T, x, y, i)
            H_xy[i, :n] = grad_x
            H_xy[i, n:] = grad_y
    
        
        for i in range(num_constraints_h2):
            grad_x, grad_y = self.problem.gradient_h_2_xy(T, x, y, i)
            H_xy[num_constraints_h1 + i, :n] = grad_x
            H_xy[num_constraints_h1 + i, n:] = grad_y
        
        mu_grad_xy = H_xy.T @ mu_vector
        
        return mu_grad_xy
    
    def compute_inner_product_T(self, T, x, y, mu_vector):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_T = np.zeros((total_constraints, n))
        
        for i in range(num_constraints_h1):
            H_T[i, :] = self.problem.gradient_h_1_T(T, x, y, i)
        
        for i in range(num_constraints_h2):
            H_T[num_constraints_h1 + i, :] = self.problem.gradient_h_2_T(T, x, y, i)
        
        mu_grad_T = H_T.T @ mu_vector
        
        return mu_grad_T

    def maxminoptimizer_g(self, T, x_init, y_init, mu_init, start_time, max_elapsed_time): # standard
        mu = mu_init.copy()
        x = x_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.maxmin_g_outer_max_iters):
            for inner_iter in range (self.maxmin_g_inner_max_iters):
                grad_x, grad_y = self.problem.gradient_g_xy(T, x, y)
                gradient_g_xy = np.concatenate([grad_x, grad_y])
                gradient_L_g_xy = gradient_g_xy + self.compute_inner_product_xy(T, x, y, mu)
                if np.linalg.norm(gradient_L_g_xy) <= self.epsilon_inner_xy_g:
                    print(f"Inner loop for L_g converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for L_g={inner_iter}, Projected Gradient norm w.r.t y of L_g={np.linalg.norm(gradient_L_g_xy)}")
                x, y = x - self.alpha_g_xy * gradient_L_g_xy[:len(x)], y - self.alpha_g_xy * gradient_L_g_xy[len(x):]
                elapsed_time = time.time() - start_time
                if elapsed_time > max_elapsed_time:
                    print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                    break
            mu_new = mu + self.beta_g_xy * self.constraint_vector(T, x, y)
            mu_new = np.maximum(mu_new, 0)
            
            
            if (np.linalg.norm(mu_new - mu) / self.beta_g_xy) <= self.epsilon_outer_xy_g:
                print(f"Outer loop for L_g converges when iter = {outer_iter}")
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
            print(f"Outer iter for L_g={outer_iter}, Projected Gradient norm w.r.t mu of L_g={(np.linalg.norm(mu_new - mu) / self.beta_g_xy)}")
            mu = mu_new
        return x, y, mu

    def maxminoptimizer_F(self, T, x, y_init, mu_init, start_time, max_elapsed_time): # standard
        mu = mu_init.copy()
        x = y_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.maxmin_F_outer_max_iters):
            for inner_iter in range (self.maxmin_g_inner_max_iters):
                grad_f_x, grad_f_y = self.problem.gradient_f_xy(T, x, y)
                gradient_f_xy = np.concatenate([grad_f_x, grad_f_y])
                grad_g_x, grad_g_y = self.problem.gradient_g_xy(T, x, y)
                gradient_g_xy = np.concatenate([grad_g_x, grad_g_y])
                # print(f"gradient_f_xy={gradient_f_xy}, gradient_g_xy={gradient_g_xy}")
                gradient_L_F_xy = gradient_f_xy + self.gamma * gradient_g_xy + self.compute_inner_product_xy(T, x, y, mu)
                if np.linalg.norm(gradient_L_F_xy) <= self.epsilon_inner_xy_F:
                    print(f"Inner loop for L_F converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for L_F={inner_iter}, Projected Gradient norm w.r.t y of L_F={np.linalg.norm(gradient_L_F_xy)}")
                x, y = x - self.alpha_F_xy * gradient_L_F_xy[:len(x)], y - self.alpha_F_xy * gradient_L_F_xy[len(x):]
                elapsed_time = time.time() - start_time
                if elapsed_time > max_elapsed_time:
                    print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                    break
            mu_new = mu + self.beta_F_xy * self.constraint_vector(T, x, y)
            mu_new = np.maximum(mu_new, 0)

            if (np.linalg.norm(mu_new - mu) / self.beta_F_xy) <= self.epsilon_outer_xy_F:
                print(f"Outer loop for L_F converges when iter = {outer_iter}")
                break
            print(f"Outer iter for L_F={outer_iter}, Projected Gradient norm w.r.t mu of L_F={(np.linalg.norm(mu_new - mu) / self.beta_F_xy)}")
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
            mu = mu_new
        return x, y, mu

    def blocc(self, T_init, x_g_init, y_g_init, y_F_init, x_F_init, mu_g_init, mu_F_init, max_elapsed_time):
        T = T_init.copy()
        x_g = x_g_init.copy()
        x_F = x_F_init.copy()
        y_g = y_g_init.copy()
        y_F = y_F_init.copy()
        mu_g = mu_g_init.copy()
        mu_F_init = mu_F_init.copy()
        history = []
        start_time = time.time()

        for iter in range (self.main_max_iters):
            print(f"Main iteration {iter + 1}")
            x_g, y_g, mu_g = self.maxminoptimizer_g(T, x_F, y_F, mu_g, start_time, max_elapsed_time)
            x_F, y_F, mu_F = self.maxminoptimizer_F(T, x_g, y_g, mu_g, start_time, max_elapsed_time)

            grad_f_T = self.problem.gradient_f_T(T, x, y_F)
            grad_g_T_xyF = self.problem.gradient_g_T(T, x_F, y_F)
            grad_g_T_xyg = self.problem.gradient_g_T(T, x_g, y_g)

            inner_product_mu_g_gradient_g_T = self.compute_inner_product_T(T, x_g, y_g, mu_g)
            inner_product_mu_F_gradient_g_T = self.compute_inner_product_T(T, x_F, y_F, mu_F)

            grad_F_T = grad_f_T + self.gamma * (grad_g_T_xyF - grad_g_T_xyg - inner_product_mu_g_gradient_g_T) + inner_product_mu_F_gradient_g_T
            grad_norm_F_T = np.linalg.norm(grad_F_T)
            if grad_norm_F_T <= self.epsilon_T:
                x = x_F
                y = y_F
                print("Main loop converged at iteration", iter)
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
            stepsize = self.alpha_T/ np.sqrt(iter + 1)
            T = T - stepsize * grad_F_T
            
            x = x_F
            y = y_F
            f_value = self.problem.f(T, x, y)
            history.append({
                'iteration': iter,
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': grad_norm_F_T,
                'time': elapsed_time
            })
            print(f"f(x,y)={f_value},grad_norm = {grad_norm_F_T}")
        return T, x, y, history
