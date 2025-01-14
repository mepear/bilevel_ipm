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
            if outer_iter == 0:
                mu_mid = mu
            else:
                mu_mid = mu + (outer_iter - 1) / (outer_iter + 2) * (mu - mu_old)

            for inner_iter in range (self.maxmin_g_inner_max_iters):
                grad_x, grad_y = self.problem.gradient_g_xy(T, x, y)
                gradient_g_xy = np.concatenate([grad_x, grad_y])
                gradient_L_g_xy = gradient_g_xy + self.compute_inner_product_xy(T, x, y, mu_mid)
                if np.linalg.norm(gradient_L_g_xy) <= self.epsilon_inner_xy_g:
                    # print(f"Inner loop for L_g converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for L_g={inner_iter}, Projected Gradient norm w.r.t y of L_g={np.linalg.norm(gradient_L_g_xy)}")
                elapsed_time = time.time() - start_time
                if elapsed_time > max_elapsed_time:
                    # print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting inner loop.")
                    return None, None, None
                stepsize = self.alpha_g_xy# / np.sqrt(inner_iter + 1)
                x, y = x - stepsize * gradient_L_g_xy[:len(x)], y - stepsize * gradient_L_g_xy[len(x):]
                elapsed_time = time.time() - start_time
                if elapsed_time > max_elapsed_time:
                    # print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                    break
            stepsize = self.beta_g_xy #/ np.sqrt(outer_iter + 1)
            # print(f"current mu is:{mu}")
            # print(f"constraints value are:{self.constraint_vector(T, x, y)}")
            mu_new = mu_mid + stepsize * self.constraint_vector(T, x, y)
            # print(f"current mu_new is:{mu_new}")
            mu_new = np.maximum(mu_new, 0)
            mu_old = mu
            # print(f"current mu_new after projection is:{mu_new}")
            
            print(f"Outer iter for L_g={outer_iter}, Projected Gradient norm w.r.t mu of L_g={(np.linalg.norm(mu_new - mu_old) / stepsize)}")
            if (np.linalg.norm(mu_new - mu_old) / stepsize) <= self.epsilon_outer_xy_g:
                print(f"Outer loop for L_g converges when iter = {outer_iter}")
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting inner loop.")
                return None, None, None
            mu = mu_new
        return x, y, mu

    def maxminoptimizer_F(self, T, x, y_init, mu_init, start_time, max_elapsed_time): # standard
        mu = mu_init.copy()
        x = y_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.maxmin_F_outer_max_iters):
            if outer_iter == 0:
                mu_mid = mu
            else:
                mu_mid = mu + (outer_iter - 1) / (outer_iter + 2) * (mu - mu_old)
            for inner_iter in range (self.maxmin_g_inner_max_iters):
                grad_f_x, grad_f_y = self.problem.gradient_f_xy(T, x, y)
                gradient_f_xy = np.concatenate([grad_f_x, grad_f_y])
                grad_g_x, grad_g_y = self.problem.gradient_g_xy(T, x, y)
                gradient_g_xy = np.concatenate([grad_g_x, grad_g_y])
                # print(f"gradient_f_xy={gradient_f_xy}, gradient_g_xy={gradient_g_xy}")
                gradient_L_F_xy = gradient_f_xy + self.gamma * gradient_g_xy + self.compute_inner_product_xy(T, x, y, mu_mid)
                if np.linalg.norm(gradient_L_F_xy) <= self.epsilon_inner_xy_F:
                    # print(f"Inner loop for L_F converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for L_F={inner_iter}, Projected Gradient norm w.r.t y of L_F={np.linalg.norm(gradient_L_F_xy)}")
                elapsed_time = time.time() - start_time
                if elapsed_time > max_elapsed_time:
                    print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting inner loop.")
                    return None, None, None 
                stepsize = self.alpha_F_xy# / np.sqrt(inner_iter + 1)
                x, y = x - stepsize * gradient_L_F_xy[:len(x)], y - stepsize * gradient_L_F_xy[len(x):]
            stepsize = self.beta_F_xy# / np.sqrt(outer_iter + 1)
            mu_new = mu_mid + stepsize * self.constraint_vector(T, x, y)
            mu_new = np.maximum(mu_new, 0)
            mu_old = mu
            print(f"Outer iter for L_F={outer_iter}, Projected Gradient norm w.r.t mu of L_F={(np.linalg.norm(mu_new - mu_old) / stepsize)}")
            if (np.linalg.norm(mu_new - mu_old) / stepsize) <= self.epsilon_outer_xy_F:
                print(f"Outer loop for L_F converges when iter = {outer_iter}")
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting inner loop.")
                return None, None, None
            mu = mu_new
        return x, y, mu

    def blocc(self, T_init, x_g_init, y_g_init, y_F_init, x_F_init, mu_g_init, mu_F_init, max_elapsed_time, step_size_type="const"):
        start_time = time.time()
        T = T_init.copy()
        x_g = x_g_init.copy()
        x_F = x_F_init.copy()
        y_g = y_g_init.copy()
        y_F = y_F_init.copy()
        mu_g = mu_g_init.copy()
        mu_F = mu_F_init.copy()
        x_g_temp, y_g_temp, mu_g_temp = self.maxminoptimizer_g(T, x_F, y_F, mu_g, start_time, max_elapsed_time)
        if x_g_temp is None and y_g_temp is None and mu_g_temp is None:
            print("Time limit exceeded in Lagrangian_g. Exiting blocc.")
        else:
            x_g, y_g, mu_g = x_g_temp, y_g_temp, mu_g_temp
        x_F_temp, y_F_temp, mu_F_temp = self.maxminoptimizer_F(T, x_g, y_g, mu_g, start_time, max_elapsed_time)
        if  x_F_temp is None and y_F_temp is None and mu_F_temp is None:
            print("Time limit exceeded in Lagrangian_g. Exiting blocc.")
        else:
            x_F, y_F, mu_F = x_F_temp, y_F_temp, mu_F_temp
        grad_f_T = self.problem.gradient_f_T(T, x_F, y_F)
        grad_g_T_xyF = self.problem.gradient_g_T(T, x_F, y_F)
        grad_g_T_xyg = self.problem.gradient_g_T(T, x_g, y_g)

        inner_product_mu_g_gradient_g_T = self.compute_inner_product_T(T, x_g, y_g, mu_g)
        inner_product_mu_F_gradient_g_T = self.compute_inner_product_T(T, x_F, y_F, mu_F)

        grad_F_T = grad_f_T + self.gamma * (grad_g_T_xyF - grad_g_T_xyg - inner_product_mu_g_gradient_g_T) + inner_product_mu_F_gradient_g_T
        grad_norm_F_T = np.linalg.norm(grad_F_T)
        
        x, y = x_F, y_F
        
        history = []
        

        for iter in range (self.main_max_iters):
            print(f"Main iteration {iter + 1}")
            x, y = x_F, y_F
            f_value = self.problem.f(T, x, y)
            elapsed_time = time.time() - start_time
            history.append({
                'iteration': iter,
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': grad_norm_F_T,
                'time': elapsed_time
            })
            print(f"f(x,y)={f_value},grad_norm = {grad_norm_F_T}")

            if grad_norm_F_T <= self.epsilon_T:
                # print("Main loop converged at iteration", iter)
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break

            if step_size_type == "const":
                T_new = T - self.alpha_T * grad_F_T
            elif step_size_type == "diminish":
                T_new = T - self.alpha_T / np.sqrt(iter + 1) * grad_F_T
            else:
                raise ValueError("step_size_type can only be 'const' or 'diminish'")
            
            x_g_temp, y_g_temp, mu_g_temp = self.maxminoptimizer_g(T, x_F, y_F, mu_g, start_time, max_elapsed_time)
            if x_g_temp is None and y_g_temp is None and mu_g_temp is None:
                print("Time limit exceeded in Lagrangian_g. Exiting blocc.")
                break
            else:
                x_g, y_g, mu_g = x_g_temp, y_g_temp, mu_g_temp
            x_F_temp, y_F_temp, mu_F_temp = self.maxminoptimizer_F(T, x_g, y_g, mu_g, start_time, max_elapsed_time)
            if  x_F_temp is None and y_F_temp is None and mu_F_temp is None:
                print("Time limit exceeded in Lagrangian_g. Exiting blocc.")
                break
            else:
                x_F, y_F, mu_F = x_F_temp, y_F_temp, mu_F_temp
            
            T = T_new
            
            grad_f_T = self.problem.gradient_f_T(T, x_F, y_F)
            grad_g_T_xyF = self.problem.gradient_g_T(T, x_F, y_F)
            grad_g_T_xyg = self.problem.gradient_g_T(T, x_g, y_g)

            inner_product_mu_g_gradient_g_T = self.compute_inner_product_T(T, x_g, y_g, mu_g)
            inner_product_mu_F_gradient_g_T = self.compute_inner_product_T(T, x_F, y_F, mu_F)

            grad_F_T = grad_f_T + self.gamma * (grad_g_T_xyF - grad_g_T_xyg - inner_product_mu_g_gradient_g_T) + inner_product_mu_F_gradient_g_T
            grad_norm_F_T = np.linalg.norm(grad_F_T)
            
            
        return T, x, y, history
