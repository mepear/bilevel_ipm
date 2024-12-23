import numpy as np
import cvxpy as cp
import time

class IGD:
    def __init__(self, problem, hparams):
        self.problem = problem
        IGD_params = hparams.get('IGD', {})

        self.M = IGD_params.get('M', 1e-4)
        self.t = IGD_params.get('t', 1.0)
        self.alpha_x = IGD_params.get('alpha_x', 1e-4)
        self.alpha_y = IGD_params.get('alpha_y', 1e-3)
        self.beta_y = IGD_params.get('beta_y', 1e-3)
        self.epsilon_x = IGD_params.get('epsilon_x', 1e-4)
        self.epsilon_y = IGD_params.get('epsilon_y', 1e-4)
        self.inner_max_iters = IGD_params.get('inner_max_iters', 100)
        self.outer_max_iters = IGD_params.get('outer_max_iters', 50)

    def inner_loop(self, x, y_init):
        y = y_init.copy()
        for iter in range(self.inner_max_iters):
            grad_y = self.problem.gradient_g_y(x, y)
            grad_norm_y = np.linalg.norm(grad_y)
            y_new = y - self.alpha_y * grad_y
            if np.linalg.norm(grad_norm_y) < self.epsilon_y:
                print("Inner loop converged at iteration", iter)
                y = y_new
                break
            y = y_new
        return y
    
    def upper_loop(self, x_init, y_init, step_size_type="const"):
        x = x_init.copy()
        y = y_init.copy()
        history = []
        start_time = time.time()

        for outer_iter in range(self.outer_max_iters):
            print(f"Outer iteration {outer_iter + 1}")
            y = self.inner_loop(x, y)
            grad_f_x = self.problem.gradient_f_x(x, y)
            grad_f_y = self.problem.gradient_f_y(x, y)
            hessian_xy = self.problem.hessian_g_xy(x, y)
            hessian_yy = self.problem.hessian_g_yy(x, y)
            try:
                v = np.linalg.solve(hessian_yy, grad_f_y)
            except np.linalg.LinAlgError:
                print("Hessian matrix is singular at outer iteration", outer_iter)
                break
            grad_F_x = grad_f_x - hessian_xy @ v
            grad_norm_x = np.linalg.norm(grad_F_x)

            if step_size_type == "const":
                x_new = x - self.alpha_x * grad_F_x
            elif step_size_type == "diminish":
                x_new = x - self.alpha_x / np.sqrt(iter + 1) * grad_F_x
            else:
                raise ValueError("step_size_type can only be 'const' or 'diminish'")
            
            elapsed_time = time.time() - start_time

            if np.linalg.norm(grad_norm_x) < self.epsilon_x:
                print("Outer loop converged at iteration", outer_iter)
                x = x_new
                break
            x = x_new
            f_value = self.problem.f(x, y)
            history.append({
                'iteration': outer_iter,
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': grad_norm_x,
                'time': elapsed_time
            })
            print(f"f(x, y) = {f_value}, grad_norm = {np.linalg.norm(grad_F_x)}")
        return x, y, history
