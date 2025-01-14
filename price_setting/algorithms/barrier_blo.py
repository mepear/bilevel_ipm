import numpy as np
import cvxpy as cp
import time

class BarrierBLO:
    def __init__(self, problem, hparams):
        self.problem = problem
        barrier_blo_params = hparams.get('barrier_blo', {})

        self.M = barrier_blo_params.get('M', 1e-4)
        self.t = barrier_blo_params.get('t', 1.0)
        self.alpha_xy = barrier_blo_params.get('alpha_xy', 1e-4)
        self.alpha_T = barrier_blo_params.get('alpha_T', 1e-3)
        self.epsilon_xy = barrier_blo_params.get('epsilon_xy', 1e-4)
        self.epsilon_T = barrier_blo_params.get('epsilon_T', 1e-4)
        self.inner_max_iters = barrier_blo_params.get('inner_max_iters', 100)
        self.outer_max_iters = barrier_blo_params.get('outer_max_iters', 50)

    def tilde_g(self, T, x, y):
        t = self.t
        barrier_h1 = np.sum([np.log(-self.problem.h_1(T, x, y, i)) for i in range(self.problem.num_constraints_h1)])
        barrier_h2 = np.sum([np.log(-self.problem.h_2(T, x, y, i)) for i in range(self.problem.num_constraints_h2)])
        return self.problem.g(T, x, y) - t * (barrier_h1 + barrier_h2)
    
    def check_constraints(self, T, x, y):
        for i in range(self.problem.num_constraints_h1):
            if self.problem.h_1(T, x, y, i) > 0:
                return False
        for i in range(self.problem.num_constraints_h2):
            if self.problem.h_2(T, x, y, i) > 0:
                return False
        return True

    def project_to_constraints(self, T, x_init, y_init):
        n = self.problem.n
        x_var = cp.Variable(n)
        y_var = cp.Variable(n)

        objective = cp.Minimize(cp.sum_squares(x_var - x_init) + cp.sum_squares(y_var - y_init))

        constraints = []
        M = self.M

        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(T, x_var, y_var, i) <= -M)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(T, x_var, y_var, i) <= -M)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        if prob.status not in ["optimal", "optimal_inaccurate"] or x_var.value is None or y_var.value is None:
            print(f"Projection failed with status: {prob.status}")
            return x_init, y_init
        return x_var.value, y_var.value
    
    def gradient_tilde_g_T(self, T, x, y):
        return self.problem.gradient_g_T(T, x, y)

    def gradient_tilde_g_xy(self, T, x, y):
        grad_g_x, grad_g_y = self.problem.gradient_g_xy(T, x, y)
        grad_x_barrier = np.zeros_like(x)
        grad_y_barrier = np.zeros_like(y)
        t = self.t

        for i in range(self.problem.num_constraints_h1):
            h_val = self.problem.h_1(T, x, y, i)
            grad_h_x, grad_h_y = self.problem.gradient_h_1_xy(T, x, y, i)
            grad_x_barrier += grad_h_x / h_val
            grad_y_barrier += grad_h_y / h_val
        for i in range(self.problem.num_constraints_h2):
            h_val = self.problem.h_2(T, x, y, i)
            grad_h_x, grad_h_y = self.problem.gradient_h_2_xy(T, x, y, i)
            grad_x_barrier += grad_h_x / h_val
            grad_y_barrier += grad_h_y / h_val

        grad_x_tilde = grad_g_x - t * grad_x_barrier
        grad_y_tilde = grad_g_y - t * grad_y_barrier
    
        return grad_x_tilde, grad_y_tilde
    
    def hessian_tilde_g_T_xy(self, T, x, y):
        return self.problem.hessian_g_T_xy(T, x, y)

    def hessian_tilde_g_xy_xy(self, T, x, y):
        t = self.t
        n = self.problem.n
        H = np.zeros((2*n, 2*n))
        for i in range(self.problem.num_constraints_h1):
            h_val = self.problem.h_1(T, x, y, i)
            grad_x_h, grad_y_h = self.problem.gradient_h_1_xy(T, x, y, i)
            grad_h = np.concatenate([grad_x_h, grad_y_h])
            H += t * np.outer(grad_h, grad_h) / (h_val**2)
        for i in range(self.problem.num_constraints_h2):
            h_val = self.problem.h_2(T, x, y, i)
            grad_x_h, grad_y_h = self.problem.gradient_h_2_xy(T, x, y, i)
            grad_h = np.concatenate([grad_x_h, grad_y_h])
            H += t * np.outer(grad_h, grad_h) / (h_val**2)
        return H
    
    def solve_constrained_g(self, T):
        n = self.problem.n
        x = cp.Variable(n)
        y = cp.Variable(n)
    
        objective = cp.Minimize((self.problem.c_1 + T) @ x + self.problem.c_2 @ y)
    
        constraints = []
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(T, x, y, i) <= 0)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(T, x, y, i) <= 0)
    
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return None, None
    
        return x.value, y.value
    
    def Interior_inner_loop(self, T): # Interior point method for original problem

        x_opt, y_opt = self.solve_constrained_g(T)
        return x_opt, y_opt

    
    def inner_loop(self, T, x_init, y_init, start_time, max_elapsed_time):  # Newton method
        x = x_init.copy()
        y = y_init.copy()
        x, y = self.project_to_constraints(T, x, y)
        for iter in range(self.inner_max_iters):
            # print(f"x,y in inner loop:{x},{y}")
            grad_x, grad_y = self.gradient_tilde_g_xy(T, x, y)
            grad_xy = np.concatenate((grad_x, grad_y))
            grad_norm_xy = np.linalg.norm(grad_xy)
            if grad_norm_xy < self.epsilon_xy:
                print(f"Inner loop converged at iteration {iter}")
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting inner loop.")
                return None, None
            Hessian_xy_xy = self.hessian_tilde_g_xy_xy(T, x, y)
            try:
                v = np.linalg.solve(Hessian_xy_xy, grad_xy)
            except np.linalg.LinAlgError:
                print("Hessian matrix is singular at outer iteration", iter)
                break
            x_new = x - self.alpha_xy * v[:self.problem.n]
            y_new = y - self.alpha_xy * v[self.problem.n:]
            if self.check_constraints(T, x_new, y_new):
                x_projected = x_new
                y_projected = y_new
            else:
                x_projected, y_projected = self.project_to_constraints(T, x_new, y_new)
            # print(f"  Gradient norm = {grad_norm_xy}")

            x = x_projected
            y = y_projected

        return x, y
    
    
    def upper_loop(self, T_init, x_init, y_init, max_elapsed_time, step_size_type="const"):
        start_time = time.time()
        T = T_init.copy()
        x = x_init.copy()
        y = y_init.copy()
        x_temp, y_temp = self.inner_loop(T, x, y, start_time, max_elapsed_time)
        if y_temp is None and x_temp is None:
            print("Time limit exceeded in Inner Loop Exiting bfbm.")
        else:
            x, y = x_temp, y_temp
        grad_f_T = self.problem.gradient_f_T(T, x, y)
        grad_f_x, grad_f_y = self.problem.gradient_f_xy(T, x, y)
        Hess_g_T_xy = self.hessian_tilde_g_T_xy(T, x, y)
        Hess_g_xy_xy = self.hessian_tilde_g_xy_xy(T, x, y)
        grad_f_xy_full = np.concatenate((grad_f_x, grad_f_y))
        v = np.linalg.solve(Hess_g_xy_xy, grad_f_xy_full)
        # print(f"x={x},y={y},v={v},grad_f_xy_full={grad_f_xy_full},Hess_g_T_xy={Hess_g_T_xy}")
        grad_F_T = grad_f_T - Hess_g_T_xy @ v
        grad_norm_T = np.linalg.norm(grad_F_T)
        history = []
        
        for outer_iter in range(self.outer_max_iters):
            f_value = self.problem.f(T, x, y)

            elapsed_time = time.time() - start_time
            history.append({
                'iteration': outer_iter,
                'T': T.copy(),
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': grad_norm_T,
                'time': elapsed_time
            })
            print(f"f(T, x, y) = {f_value}, grad_norm of hyperfunction= {grad_norm_T}")

            if grad_norm_T < self.epsilon_T:
                # print("Outer loop converged at iteration", outer_iter)
                break
            
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                # print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break

            if step_size_type == "const":
                step_size = self.alpha_T
            elif step_size_type == "diminish":
                step_size = self.alpha_T / np.sqrt(outer_iter + 1)
            else:
                raise ValueError("step_size_type can only be 'const' or 'diminish'")
            
            T_new = T - step_size * grad_F_T

            x_temp, y_temp = self.inner_loop(T, x, y, start_time, max_elapsed_time)
            if y_temp is None and x_temp is None:
                print("Time limit exceeded in Inner Loop Exiting bfbm.")
                break
            else:
                x, y = x_temp, y_temp

            T = T_new

            grad_f_T = self.problem.gradient_f_T(T, x, y)
            grad_f_x, grad_f_y = self.problem.gradient_f_xy(T, x, y)
            Hess_g_T_xy = self.hessian_tilde_g_T_xy(T, x, y)
            Hess_g_xy_xy = self.hessian_tilde_g_xy_xy(T, x, y)
            grad_f_xy_full = np.concatenate((grad_f_x, grad_f_y))
            v = np.linalg.solve(Hess_g_xy_xy, grad_f_xy_full)
            # print(f"x={x},y={y},v={v},grad_f_xy_full={grad_f_xy_full},Hess_g_T_xy={Hess_g_T_xy}")
            grad_F_T = grad_f_T - Hess_g_T_xy @ v
            grad_norm_T = np.linalg.norm(grad_F_T)
            
            
            
            
            
            
        return T, x, y, history
