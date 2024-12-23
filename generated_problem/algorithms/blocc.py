# BLOCC.py

import numpy as np
import cvxpy as cp
import time

class BLOCC:
    def __init__(self, problem, hparams):
        self.problem = problem
        blocc_params = hparams.get('blocc', {})

        self.alpha_x = blocc_params.get('alpha_x', 1e-4)
        self.gamma = blocc_params.get('gamma', 0.1)
        self.alpha_g_y = blocc_params.get('alpha_g_y', 0.1)
        self.alpha_F_y = blocc_params.get('alpha_F_y', 0.1)
        self.beta_g_y = blocc_params.get('beta_g_y', 0.1)
        self.beta_F_y = blocc_params.get('beta_F_y', 0.1)
        self.epsilon_x = blocc_params.get('epsilon_x', 1e-6)
        self.epsilon_inner_y_g = blocc_params.get('epsilon_inner_y_g', 1e-6)
        self.epsilon_outer_y_g = blocc_params.get('epsilon_outer_y_g', 1e-6)
        self.epsilon_inner_y_F = blocc_params.get('epsilon_inner_y_F', 1e-6)
        self.epsilon_outer_y_F = blocc_params.get('epsilon_outer_y_F', 1e-6)
        self.maxmin_g_outer_max_iters = blocc_params.get('maxmin_g_outer_max_iters', 10000)
        self.maxmin_F_outer_max_iters = blocc_params.get('maxmin_F_outer_max_iters', 10000)
        self.maxmin_g_inner_max_iters = blocc_params.get('maxmin_g_inner_max_iters', 10000)
        self.maxmin_F_inner_max_iters = blocc_params.get('maxmin_F_inner_max_iters', 10000)
        self.main_max_iters = blocc_params.get('main_max_iters', 10000)
    
    def constraint_vector(self, x, y):
        constraints = []
    
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i))
    
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i))
    
        constraint_vector = np.array(constraints)
        return constraint_vector

    def compute_inner_product_y(self, x, y, mu_vector):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_y = np.zeros((total_constraints, n))
        
        for i in range(num_constraints_h1):
            H_y[i, :] = self.problem.gradient_h_1_y(x, y, i)
        
        for i in range(num_constraints_h2):
            H_y[num_constraints_h1 + i, :] = self.problem.gradient_h_2_y(x, y, i)
        
        mu_grad_y = H_y.T @ mu_vector
        
        return mu_grad_y
    
    def compute_inner_product_x(self, x, y, mu_vector):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_x = np.zeros((total_constraints, n))
        
        for i in range(num_constraints_h1):
            H_x[i, :] = self.problem.gradient_h_1_x(x, y, i)
        
        for i in range(num_constraints_h2):
            H_x[num_constraints_h1 + i, :] = self.problem.gradient_h_2_x(x, y, i)
        
        mu_grad_x = H_x.T @ mu_vector
        
        return mu_grad_x
    
    def project_to_constraints(self, x, y_init):
        n = self.problem.n
        y = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(y - y_init))
        constraints = []

        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= 0)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        return y.value
    
    def maxminoptimizer_g(self, x, y_init, mu_init, start_time, max_elapsed_time): # acceleterated
        mu = mu_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.maxmin_g_outer_max_iters):
            if outer_iter == 0:
                mu_mid = mu
            else:
                mu_mid = mu + (outer_iter - 1) / (outer_iter + 2) * (mu - mu_old)

            for inner_iter in range (self.maxmin_g_inner_max_iters):
                gradient_L_g_y = self.problem.gradient_g_y(x,y) + self.compute_inner_product_y(x, y, mu_mid)
                # y = y - self.alpha_g_y * gradient_L_g_y
                if np.linalg.norm(gradient_L_g_y) <= self.epsilon_inner_y_g:
                    # print(f"Inner loop for L_g converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for L_g={inner_iter}, Projected Gradient norm w.r.t y of L_g={np.linalg.norm(gradient_L_g_y)}")
                y = y - self.alpha_g_y * gradient_L_g_y
                # elapsed_time = time.time() - start_time
                # if elapsed_time > max_elapsed_time:
                #     print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                #     break
                
            mu_new = mu_mid + self.beta_g_y * self.constraint_vector(x, y)
            mu_new = np.maximum(mu_new, 0)
            mu_old = mu
            
            
            if (np.linalg.norm(mu_new - mu_mid) / self.beta_g_y) <= self.epsilon_outer_y_g:
                # print(f"Outer loop for L_g converges when iter = {outer_iter}")
                break
            # elapsed_time = time.time() - start_time
            # if elapsed_time > max_elapsed_time:
            #     print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
            #     break  
            # print(f"Outer iter for L_g={outer_iter}, Projected Gradient norm w.r.t mu of L_g={(np.linalg.norm(mu_new - mu_old) / self.beta_g_y)}")
            mu = mu_new
        return y, mu

    # def maxminoptimizer_g(self, x, y_init, mu_init, start_time, max_elapsed_time): # standard
    #     mu = mu_init.copy()
    #     y = y_init.copy()
    #     for outer_iter in range (self.maxmin_g_outer_max_iters):
    #         for inner_iter in range (self.maxmin_g_inner_max_iters):
    #             gradient_L_g_y = self.problem.gradient_g_y(x,y) + self.compute_inner_product_y(x, y, mu)
    #             if np.linalg.norm(gradient_L_g_y) <= self.epsilon_inner_y_g:
    #                 # print(f"Inner loop for L_g converges when iter = {inner_iter}")
    #                 break
    #             # print(f"Inner iter for L_g={inner_iter}, Projected Gradient norm w.r.t y of L_g={np.linalg.norm(gradient_L_g_y)}")
    #             y = y - self.alpha_g_y * gradient_L_g_y
    #             elapsed_time = time.time() - start_time
    #             if elapsed_time > max_elapsed_time:
    #                 print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
    #                 break
    #         mu_new = mu + self.beta_g_y * self.constraint_vector(x, y)
    #         mu_new = np.maximum(mu_new, 0)
            
            
    #         if (np.linalg.norm(mu_new - mu) / self.beta_g_y) <= self.epsilon_outer_y_g:
    #             print(f"Outer loop for L_g converges when iter = {outer_iter}")
    #             break
    #         elapsed_time = time.time() - start_time
    #         if elapsed_time > max_elapsed_time:
    #             print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
    #             break
    #         # print(f"Outer iter for L_g={outer_iter}, Projected Gradient norm w.r.t mu of L_g={(np.linalg.norm(mu_new - mu) / self.beta_g_y)}")
    #         mu = mu_new
    #     return y, mu
    
    def maxminoptimizer_F(self, x, y_init, mu_init, start_time, max_elapsed_time): # acceleterated
        mu = mu_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.maxmin_F_outer_max_iters):
            if outer_iter == 0:
                mu_mid = mu
            else:
                mu_mid = mu + (outer_iter - 1) / (outer_iter + 2) * (mu - mu_old)
            
            for inner_iter in range (self.maxmin_g_inner_max_iters):
                gradient_L_F_y = self.problem.gradient_f_y(x,y) + self.gamma * self.problem.gradient_g_y(x,y) + self.compute_inner_product_y(x, y, mu_mid)
                # y = y - self.alpha_F_y * gradient_L_F_y
                if np.linalg.norm(gradient_L_F_y) <= self.epsilon_inner_y_F:
                    # print(f"Inner loop for L_F converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for L_F={inner_iter}, Projected Gradient norm w.r.t y of L_F={np.linalg.norm(gradient_L_F_y)}")
                y = y - self.alpha_F_y * gradient_L_F_y
                # elapsed_time = time.time() - start_time
                # if elapsed_time > max_elapsed_time:
                #     print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                #     break
             
            mu_new = mu_mid + self.beta_F_y * self.constraint_vector(x, y)
            mu_new = np.maximum(mu_new, 0)
            mu_old = mu

            if (np.linalg.norm(mu_new - mu_mid) / self.beta_F_y) <= self.epsilon_outer_y_F:
                # print(f"Outer loop for L_F converges when iter = {outer_iter}")
                break
            # elapsed_time = time.time() - start_time
            # if elapsed_time > max_elapsed_time:
            #     print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
            #     break  
            # print(f"Outer iter for L_F={outer_iter}, Projected Gradient norm w.r.t mu of L_F={(np.linalg.norm(mu_new - mu_old) / self.beta_F_y)}")
            mu = mu_new
        return y, mu

    # def maxminoptimizer_F(self, x, y_init, mu_init, start_time, max_elapsed_time): # standard
    #     mu = mu_init.copy()
    #     y = y_init.copy()
    #     for outer_iter in range (self.maxmin_F_outer_max_iters):
    #         for inner_iter in range (self.maxmin_g_inner_max_iters):
    #             gradient_L_F_y = self.problem.gradient_f_y(x,y) + self.gamma * self.problem.gradient_g_y(x,y) + self.compute_inner_product_y(x, y, mu)
    #             if np.linalg.norm(gradient_L_F_y) <= self.epsilon_inner_y_F:
    #                 # print(f"Inner loop for L_F converges when iter = {inner_iter}")
    #                 break
    #             # print(f"Inner iter for L_F={inner_iter}, Projected Gradient norm w.r.t y of L_F={np.linalg.norm(gradient_L_F_y)}")
    #             y = y - self.alpha_F_y * gradient_L_F_y
    #             elapsed_time = time.time() - start_time
    #             if elapsed_time > max_elapsed_time:
    #                 print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
    #                 break
    #         mu_new = mu + self.beta_F_y * self.constraint_vector(x, y)
    #         mu_new = np.maximum(mu_new, 0)

    #         if (np.linalg.norm(mu_new - mu) / self.beta_F_y) <= self.epsilon_outer_y_F:
    #             print(f"Outer loop for L_F converges when iter = {outer_iter}")
    #             break
    #         # print(f"Outer iter for L_F={outer_iter}, Projected Gradient norm w.r.t mu of L_F={(np.linalg.norm(mu_new - mu) / self.beta_F_y)}")
    #         elapsed_time = time.time() - start_time
    #         if elapsed_time > max_elapsed_time:
    #             print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
    #             break
    #         mu = mu_new
    #     return y, mu

    def blocc(self, x_init, y_g_init, y_F_init, mu_g_init, mu_F_init, max_elapsed_time, step_size_type="const"):
        x = x_init.copy()
        y_g = y_g_init.copy()
        y_F = y_F_init.copy()
        mu_g = mu_g_init.copy()
        mu_F_init = mu_F_init.copy()
        history = []
        start_time = time.time()

        for iter in range (self.main_max_iters):
            print(f"Main iteration {iter + 1}")
            y_g, mu_g = self.maxminoptimizer_g(x, y_F, mu_g, start_time, max_elapsed_time)
            y_F, mu_F = self.maxminoptimizer_F(x, y_g, mu_g, start_time, max_elapsed_time)

            grad_f_x = self.problem.gradient_f_x(x, y_F)
            grad_g_x_yF = self.problem.gradient_g_x(x, y_F)
            grad_g_x_yg = self.problem.gradient_g_x(x, y_g)

            inner_product_mu_g_gradient_g_x = self.compute_inner_product_x(x, y_g, mu_g)
            inner_product_mu_F_gradient_g_x = self.compute_inner_product_x(x, y_F, mu_F)

            grad_F_x = grad_f_x + self.gamma * (grad_g_x_yF - grad_g_x_yg - inner_product_mu_g_gradient_g_x) + inner_product_mu_F_gradient_g_x
            grad_norm_F_x = np.linalg.norm(grad_F_x)
            if grad_norm_F_x <= self.epsilon_x:
                y = y_F
                print("Main loop converged at iteration", iter)
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
            
            if step_size_type == "const":
                x = x - self.alpha_x * grad_F_x
            elif step_size_type == "diminish":
                x = x - self.alpha_x / np.sqrt(iter + 1) * grad_F_x
            else:
                raise ValueError("step_size_type can only be 'const' or 'diminish'")
            
            y = y_F
            f_value = self.problem.f(x, y)
            history.append({
                'iteration': iter,
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': grad_norm_F_x,
                'time': elapsed_time
            })
            print(f"f(x,y)={f_value},grad_norm = {grad_norm_F_x}")
        return x, y, history
