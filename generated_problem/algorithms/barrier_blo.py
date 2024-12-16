import numpy as np
import cvxpy as cp
import time

class BarrierBLO:
    def __init__(self, problem, hparams):
        self.problem = problem
        barrier_blo_params = hparams.get('barrier_blo', {})

        self.M = barrier_blo_params.get('M', 1e-4)
        self.t = barrier_blo_params.get('t', 1.0)
        self.alpha_x = barrier_blo_params.get('alpha_x', 1e-4)
        self.alpha_y = barrier_blo_params.get('alpha_y', 1e-3)
        self.beta_y = barrier_blo_params.get('beta_y', 1e-3)
        self.epsilon_x = barrier_blo_params.get('epsilon_x', 1e-4)
        self.epsilon_y = barrier_blo_params.get('epsilon_y', 1e-4)
        self.inner_max_iters = barrier_blo_params.get('inner_max_iters', 100)
        self.outer_max_iters = barrier_blo_params.get('outer_max_iters', 50)
    
    # def tilde_g(self, x, y): # CP
    #     t = self.t
    #     barrier_h1 = cp.sum([cp.log(-self.problem.h_1(x, y, i)) for i in range(self.problem.num_constraints_h1)])
    #     barrier_h2 = cp.sum([cp.log(-self.problem.h_2(x, y, i)) for i in range(self.problem.num_constraints_h2)])
    #     # barrier_h3 = cp.log(-self.problem.h_3(x, y))
    #     tilde_g_expr = self.problem.g(x, y) - t * (barrier_h1 + barrier_h2)
    #     # tilde_g_expr = self.problem.g(x, y) - t * (barrier_h1 + barrier_h3)
    #     return tilde_g_expr

    def tilde_g(self, x, y): # NP
        t = self.t
        barrier_h1 = np.sum([np.log(-self.problem.h_1(x, y, i)) for i in range(self.problem.num_constraints_h1)])
        barrier_h2 = np.sum([np.log(-self.problem.h_2(x, y, i)) for i in range(self.problem.num_constraints_h2)])
        return self.problem.g(x, y) - t * (barrier_h1 + barrier_h2)
    
    def check_constraints(self, x, y):
        for i in range(self.problem.num_constraints_h1):
            if self.problem.h_1(x, y, i) > 0:
                return False

        for i in range(self.problem.num_constraints_h2):
            if self.problem.h_2(x, y, i) > 0:
                return False
        return True

    def project_to_constraints(self, x, y_init):
        n = self.problem.n
        y = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(y - y_init))
        constraints = []
        M = self.M

        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= -M)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= -M)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        if prob.status not in ["optimal", "optimal_inaccurate"] or y.value is None:
            # Handle the error, for example:
            print(f"Projection failed with status: {prob.status}")
            return y_init  # or some fallback strategy
        return y.value
    
    def gradient_tilde_g_x(self, x, y):
        grad_g_x = self.problem.gradient_g_x(x, y)
        sum_grad_hx_over_hi = np.zeros_like(x)
        t = self.t
        for i in range(self.problem.num_constraints_h1):
            hi = self.problem.h_1(x, y, i)
            grad_hi_x = self.problem.gradient_h_1_x(x, y, i)
            sum_grad_hx_over_hi += grad_hi_x / hi
        for i in range(self.problem.num_constraints_h2):
            hi = self.problem.h_2(x, y, i)
            grad_hi_x = self.problem.gradient_h_2_x(x, y, i)
            sum_grad_hx_over_hi += grad_hi_x / hi
        grad_tilde_g_x = grad_g_x - t * sum_grad_hx_over_hi
        return grad_tilde_g_x

    def gradient_tilde_g_y(self, x, y):
        grad_g_y = self.problem.gradient_g_y(x, y)
        sum_grad_hy_over_hi = np.zeros_like(y)
        t = self.t
        for i in range(self.problem.num_constraints_h1):
            hi = self.problem.h_1(x, y, i)
            grad_hi_y = self.problem.gradient_h_1_y(x, y, i)
            sum_grad_hy_over_hi += grad_hi_y / hi
        for i in range(self.problem.num_constraints_h2):
            hi = self.problem.h_2(x, y, i)
            grad_hi_y = self.problem.gradient_h_2_y(x, y, i)
            sum_grad_hy_over_hi += grad_hi_y / hi
        grad_tilde_g_y = grad_g_y - t * sum_grad_hy_over_hi
        # h_3 = self.problem.h_3(x, y)
        # grad_tilde_g_y = grad_g_y - t * sum_grad_hy_over_hi - t * self.problem.gradient_h_3_y(x, y) / h_3
        return grad_tilde_g_y

    def hessian_tilde_g_xy(self, x, y):
        hessian_g_xy = self.problem.hessian_g_xy(x, y)
        sum_constraints = np.zeros((self.problem.n, self.problem.n))
        t = self.t
        for i in range(self.problem.num_constraints_h1):
            hi = self.problem.h_1(x, y, i)
            grad_hi_x = self.problem.gradient_h_1_x(x, y, i)
            grad_hi_y = self.problem.gradient_h_1_y(x, y, i)
            hessian_hi_xy = self.problem.hessian_h_1_xy(x, y, i)
            term = (hessian_hi_xy / hi) - (np.outer(grad_hi_x, grad_hi_y) / hi**2)
            sum_constraints += term
        hessian_tilde_g_xy = hessian_g_xy - t * sum_constraints
        return hessian_tilde_g_xy

    def hessian_tilde_g_yy(self, x, y):
        hessian_g_yy = self.problem.hessian_g_yy(x, y)
        sum_constraints = np.zeros((self.problem.n, self.problem.n))
        t = self.t
        for i in range(self.problem.num_constraints_h1):
            hi = self.problem.h_1(x, y, i)
            grad_hi_y = self.problem.gradient_h_1_y(x, y, i)
            hessian_hi_yy = self.problem.hessian_h_1_yy(x, y, i)
            term = (hessian_hi_yy / hi) - (np.outer(grad_hi_y, grad_hi_y) / hi**2)
            sum_constraints += term
        for i in range(self.problem.num_constraints_h2):
            hi = self.problem.h_2(x, y, i)
            grad_hi_y = self.problem.gradient_h_2_y(x, y, i)
            hessian_hi_yy = self.problem.hessian_h_2_yy(x, y, i)
            term = (hessian_hi_yy / hi) - (np.outer(grad_hi_y, grad_hi_y) / hi**2)
            sum_constraints += term
        hessian_tilde_g_yy = hessian_g_yy - t * sum_constraints
        # h3 = self.problem.h_3(x, y)
        # grad_h3_y = self.problem.gradient_h_3_y(x, y)
        # hessian_h3_yy = self.problem.hessian_h_3_yy(x, y)
        # term2 = (hessian_h3_yy / h3) - (np.outer(grad_h3_y, grad_h3_y) / h3**2)
        # hessian_tilde_g_yy = hessian_g_yy - t * sum_constraints - t * term2
        return hessian_tilde_g_yy
    
    def solve_constrained_g(self, x): # Use solver to solve constrained original problem
        n = self.problem.n
        y = cp.Variable(n)
        
        objective = cp.Minimize(self.problem.g(x, y))
        
        constraints = []
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= 0)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= 0)
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return y.value
        
        return y.value
    
    def solve_unconstrained_tilde_g(self, x): # Use solver to solve unconstrained barrier reformulated problem
        n = self.problem.n
        y = cp.Variable(n)

        tilde_g = self.tilde_g(x, y)
        objective = cp.Minimize(tilde_g)

        prob = cp.Problem(objective)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return y.value

        return y.value
    
    def solve_with_interior_point(self, x): # Use interior point method to solve constrained barrier reformulated problem
        n = self.problem.n
        y = cp.Variable(n)
        
        objective = cp.Minimize(self.tilde_g(x, y))
        
        constraints = []
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= -self.M)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= -self.M)
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=True)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return None
        
        return y.value

    def solve_constrained_tilde_g(self, x): # Use convex solver to solve constrained barrier reformulated problem
        n = self.problem.n
        y = cp.Variable(n)
        
        objective = cp.Minimize(self.tilde_g(x, y))
        
        constraints = []
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= 0)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= 0)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return y.value
        
        return y.value
    
    def Interior_inner_loop(self, x, y_init): # Interior point method for original problem

        y_opt = self.solve_constrained_g(x)
        return y_opt
    
    # def compute_step_size(self, y, d, x):
    #     t_max = np.inf
    #     alpha = 0.99  

    #     for i in range(self.problem.num_constraints_h1):
    #         a_i = self.problem.gradient_h_1_y(x, y, i)
    #         b_i = self.problem.h_1(x, y, i)
    #         a_dot_d = np.dot(a_i, d)
            
    #         if a_dot_d > 1e-12:
    #             t_i = -b_i / a_dot_d
    #             if t_i < t_max:
    #                 t_max = t_i
        
    #     for i in range(self.problem.num_constraints_h2):
    #         a_i = self.problem.gradient_h_2_y(x, y, i)
    #         b_i = self.problem.h_2(x, y, i)
    #         a_dot_d = np.dot(a_i, d)
            
    #         if a_dot_d > 1e-12:
    #             t_i = -b_i / a_dot_d
    #             if t_i < t_max:
    #                 t_max = t_i
        
    #     t_max = alpha * t_max
        
    #     if t_max <= 0:
    #         raise ValueError("Fail to find the stepsize.")
        
    #     return t_max

    # def inner_loop(self, x, y_init): # Use solver

    #     y_opt = self.solve_unconstrained_tilde_g(x)
    #     grad_y = self.gradient_tilde_g_y(x, y_opt)
    #     grad_norm_y = np.linalg.norm(grad_y)
    #     print(f"gradient norm of g={grad_norm_y}")
    #     return y_opt

    # def inner_loop(self, x, y_init): # Interior point method for barrier reformulated problem

    #     y_opt = self.solve_with_interior_point(x)
    #     grad_y = self.gradient_tilde_g_y(x, y_opt)
    #     grad_norm_y = np.linalg.norm(grad_y)
    #     print(f"gradient norm of g={grad_norm_y}")
    #     return y_opt

    # def inner_loop(self, x, y_init): # Accelerated PGD method
    #     y = y_init.copy()
    #     y = self.project_to_constraints(x, y)
    #     for iter in range(self.inner_max_iters):
    #         if iter == 0:
    #             w = y
    #         else:
    #             w = y + (self.beta_y / (iter + 1)) * (y - y_old)
    #         y_old = y
    #         grad_w = self.gradient_tilde_g_y(x, w)
    #         y_new = y - (self.alpha_y / (iter + 1)) * grad_w
    #         y_projected = self.project_to_constraints(x, y_new)
    #         grad_y = self.gradient_tilde_g_y(x, y_projected)
    #         grad_norm_y = np.linalg.norm(grad_y)
    #         print(f"Iterations={iter}, gradient norm={grad_norm_y}")
    #         if np.linalg.norm(grad_norm_y) < self.epsilon_y:
    #             print("Inner loop converged at iteration", iter)
    #             y = y_projected
    #             break
    #         y = y_projected
    #     return y

    # def inner_loop(self, x, y_init):  # Standard PGD method
    #     y = y_init.copy()
    #     for iter in range(self.inner_max_iters):
    #         grad_y = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y = np.linalg.norm(grad_y)
    #         stepsize = self.alpha_y / ((iter + 1) * grad_norm_y)
    #         y_new = y - stepsize * grad_y
    #         # print(f"y_new={y_new}")
    #         if self.check_constraints(x, y_new):
    #             y_projected = y_new
    #             # print("  Constraints satisfied. No projection needed.")
    #         else:
    #             y_projected = self.project_to_constraints(x, y_new)
    #             # print("  Constraints violated. Projection performed.")
    #         # print(f"y_projected={y_projected}")
    #         print(f"  Gradient norm = {grad_norm_y}")

    #         if grad_norm_y < self.epsilon_y:
    #             print(f"Inner loop converged at iteration {iter}")
    #             y = y_projected
    #             break

    #         y = y_projected

    #     return y
    
    # def inner_loop(self, x, y_init):  # Use specific stepsize
    #     y = y_init.copy()
    #     for iter in range(self.inner_max_iters):
    #         grad_y = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y = np.linalg.norm(grad_y)
    #         stepsize = np.minimum(self.alpha_y, self.compute_step_size(y, -grad_y, x)) 
    #         y_new = y - stepsize * grad_y
    #         # print(f"y_new={y_new}")
    #         # if self.check_constraints(x, y_new):
    #         #     y_projected = y_new
    #             # print("  Constraints satisfied. No projection needed.")
    #         # else:
    #         #     y_projected = self.project_to_constraints(x, y_new)
    #             # print("  Constraints violated. Projection performed.")
    #         # print(f"y_projected={y_projected}")
    #         print(f"  Gradient norm = {grad_norm_y}")

    #         if grad_norm_y < self.epsilon_y:
    #             print(f"Inner loop converged at iteration {iter}")
    #             y = y_new
    #             break

    #         y = y_new

    #     return y
    
    def inner_loop(self, x, y_init):  # Newton method
        y = y_init.copy()
        for iter in range(self.inner_max_iters):
            grad_y = self.gradient_tilde_g_y(x, y)
            grad_norm_y = np.linalg.norm(grad_y)
            if grad_norm_y < self.epsilon_y:
                print(f"Inner loop converged at iteration {iter}")
                break
            Hessian_yy = self.hessian_tilde_g_yy(x, y)
            try:
                v = np.linalg.solve(Hessian_yy, grad_y)
            except np.linalg.LinAlgError:
                print("Hessian matrix is singular at outer iteration", iter)
                break
            y_new = y - (self.alpha_y) * v
            if self.check_constraints(x, y_new):
                y_projected = y_new
            else:
                y_projected = self.project_to_constraints(x, y_new)
            # print(f"  Gradient norm = {grad_norm_y}")

            y = y_projected

        return y
    
    # def inner_loop(self, x, y_init):  # BFGS method
    #     y = y_init.copy()
    #     n = len(y)
    #     H = np.eye(n)  # Initialize inverse Hessian approximation as identity matrix

    #     grad_y = self.gradient_tilde_g_y(x, y)
    #     grad_norm_y = np.linalg.norm(grad_y)

    #     for iter in range(self.inner_max_iters):
    #         p = -H @ grad_y

    #         step_size = self.alpha_y / (iter + 1)

    #         y_new = y + step_size * p

    #         if self.check_constraints(x, y_new):
    #             y_projected = y_new
    #         else:
    #             y_projected = self.project_to_constraints(x, y_new)

    #         s = y_projected - y  

    #         y = y_projected

    #         grad_y_new = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y_new = np.linalg.norm(grad_y_new)

    #         y_diff = s
    #         grad_diff = grad_y_new - grad_y  

    #         sTy = grad_diff.T @ y_diff
    #         if np.abs(sTy) < 1e-10:
    #             print(f"Warning: Division by zero in BFGS update at iteration {iter}")
    #             break

    #         rho = 1.0 / sTy

    #         I = np.eye(n)
    #         H = (I - rho * np.outer(y_diff, grad_diff)) @ H @ (I - rho * np.outer(grad_diff, y_diff)) + rho * np.outer(y_diff, y_diff)

    #         grad_y = grad_y_new
    #         grad_norm_y = grad_norm_y_new

    #         if grad_norm_y < self.epsilon_y:
    #             print(f"Inner loop converged at iteration {iter}")
    #             break

    #     return y
    
    # def inner_loop(self, x, y_init): # Mixed strategy (PGD and BFGS)
    #     y = y_init.copy()
    #     n = len(y)
    #     H = np.eye(n)

    #     grad_y = self.gradient_tilde_g_y(x, y)
    #     grad_norm_y = np.linalg.norm(grad_y)

    #     for iter in range(self.inner_max_iters):
    #         if grad_norm_y >= 50:
    #             stepsize = self.beta_y / ((iter + 1) * grad_norm_y)
    #             y_new = y - stepsize * grad_y
    #             method = "Gradient Descent"
    #         else:
    #             if iter == 0 or 's' not in locals():
    #                 H = np.eye(n)
    #             else:
    #                 s = y - y_prev  # s = y_k - y_{k-1}
    #                 grad_diff = grad_y - grad_y_prev  # y_diff = grad_y_k - grad_y_{k-1}
    #                 sTy = grad_diff.T @ s
    #                 if np.abs(sTy) > 1e-10:
    #                     rho = 1.0 / sTy
    #                     I = np.eye(n)
    #                     H = (I - rho * np.outer(s, grad_diff)) @ H @ (I - rho * np.outer(grad_diff, s)) + rho * np.outer(s, s)
    #                 else:
    #                     print(f"Warning: Division by zero in BFGS update at iteration {iter}")
    #                     H = np.eye(n)

    #             p = -H @ grad_y
    #             stepsize = self.alpha_y / (iter + 1)
    #             y_new = y + stepsize * p
    #             method = "BFGS"

    #         if self.check_constraints(x, y_new):
    #             y_projected = y_new
    #         else:
    #             y_projected = self.project_to_constraints(x, y_new)

    #         print(f"Iteration {iter}, Method: {method}, Gradient norm = {grad_norm_y}")

    #         if grad_norm_y < self.epsilon_y:
    #             print(f"Inner loop converged at iteration {iter}")
    #             y = y_projected
    #             break

    #         if grad_norm_y < 50 and method == "BFGS":
    #             y_prev = y.copy()
    #             grad_y_prev = grad_y.copy()

    #         y = y_projected.copy()
    #         grad_y = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y = np.linalg.norm(grad_y)

    #     return y

    # def inner_loop(self, x, y_init): # Mixed strategy (PGD and Newton)
    #     y = y_init.copy()
    #     n = len(y)

    #     for iter in range(self.inner_max_iters):
    #         grad_y = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y = np.linalg.norm(grad_y)
        
    #         if grad_norm_y >= 50:
    #             stepsize = self.alpha_y / ((iter + 1) * grad_norm_y)
    #             y_new = y - stepsize * grad_y
    #             method = "Projected Gradient Descent"
    #         else:
    #             Hessian_yy = self.hessian_tilde_g_yy(x, y)
    #             try:
    #                 p = -np.linalg.solve(Hessian_yy, grad_y)
    #                 y_new = y + p 
    #                 method = "Newton's Method"
    #             except np.linalg.LinAlgError:
    #                 print(f"Hessian matrix is singular at iteration {iter}. Falling back to gradient descent.")
    #                 stepsize = self.alpha_y / ((iter + 1) * grad_norm_y)
    #                 y_new = y - stepsize * grad_y
    #                 method = "Projected Gradient Descent (Fallback)"

    #         if self.check_constraints(x, y_new):
    #             y_projected = y_new
    #         else:
    #             y_projected = self.project_to_constraints(x, y_new)

    #         print(f"Iteration {iter}, Method: {method}, Gradient norm = {grad_norm_y}")

    #         if grad_norm_y < self.epsilon_y:
    #             print(f"Inner loop converged at iteration {iter}")
    #             y = y_projected
    #             break

    #         y = y_projected.copy()
        
    #     return y
    
    # def inner_loop(self, x, y_init):  # Newton method with Line Search
    #     y = y_init.copy()
    #     for iter in range(self.inner_max_iters):
    #         y = self.project_to_constraints(x, y)
    #         grad_y = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y = np.linalg.norm(grad_y)
    #         Hessian_yy = self.hessian_tilde_g_yy(x, y)
    #         try:
    #             v = np.linalg.solve(Hessian_yy, grad_y)
    #         except np.linalg.LinAlgError:
    #             print("Hessian matrix is singular at iteration", iter)
    #             break
        
    #         alpha = 1.0  
    #         rho = 0.5    
    #         c = 1e-1     
        
    #         while True:
    #             y_new = y - alpha * v
    #             if self.check_constraints(x, y_new):
    #                 y_projected = y_new
    #             else:
    #                 y_projected = self.project_to_constraints(x, y_new)
                
    #             f_current = self.tilde_g(x, y)
    #             f_new = self.tilde_g(x, y_projected)
    #             lhs = f_new
    #             rhs = f_current - c * alpha * grad_y.T @ v
    #             print(f"When alpha={alpha}, lhs={lhs}, rhs={rhs}, iteraion={iter}, gradient norm={grad_norm_y}")
    #             if lhs <= rhs:
    #                 break
    #             else:
    #                 alpha *= rho
    #                 if alpha < 1e-8:
    #                     print("Line search failed to find a suitable step size at iteration", iter)
    #                     break
        
    #         # print(f"Iteration {iter}, Step size = {alpha}, Gradient norm = {grad_norm_y}")
        
    #         if grad_norm_y < self.epsilon_y:
    #             print(f"Inner loop converged at iteration {iter}")
    #             y = y_projected
    #             break
        
    #         y = y_projected.copy()
        
    #     return y


    def upper_loop(self, x_init, y_init, max_elapsed_time):
        x = x_init.copy()
        y = y_init.copy()
        history = []
        start_time = time.time()

        for outer_iter in range(self.outer_max_iters):
            print(f"Outer iteration {outer_iter + 1}")
            y = self.inner_loop(x, y)
            grad_f_x = self.problem.gradient_f_x(x, y)
            grad_f_y = self.problem.gradient_f_y(x, y)
            hessian_xy = self.hessian_tilde_g_xy(x, y)
            hessian_yy = self.hessian_tilde_g_yy(x, y)
            try:
                v = np.linalg.solve(hessian_yy, grad_f_y)
            except np.linalg.LinAlgError:
                print("Hessian matrix is singular at outer iteration", outer_iter)
                break
            grad_F_x = grad_f_x - hessian_xy @ v
            grad_norm_x = np.linalg.norm(grad_F_x)
            if np.linalg.norm(grad_norm_x) < self.epsilon_x:
                print("Outer loop converged at iteration", outer_iter)
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
            step_size = self.alpha_x / np.sqrt(outer_iter + 1)
            # step_size = self.alpha_x / np.sqrt(outer_iter + 1)
            x_new = x - step_size * grad_F_x
            # x_new = x - self.alpha_x * grad_F_x
            elapsed_time = time.time() - start_time    
            
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
            print(f"f(x, y) = {f_value}, grad_norm of hyperfunction= {np.linalg.norm(grad_F_x)}")
            
        return x, y, history
