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
    
    def tilde_g(self, x, y):
        t = self.t
        barrier_h1 = cp.sum([cp.log(-self.problem.h_1(x, y, i)) for i in range(self.problem.num_constraints_h1)])
        barrier_h2 = cp.sum([cp.log(-self.problem.h_2(x, y, i)) for i in range(self.problem.num_constraints_h2)])
        barrier_h3 = cp.log(-self.problem.h_3(x, y))
        # tilde_g_expr = self.problem.g(x, y) - t * (barrier_h1 + barrier_h2)
        tilde_g_expr = self.problem.g(x, y) - t * (barrier_h1 + barrier_h3)
        return tilde_g_expr

    # def tilde_g(self, x, y):
    #     t = self.t
    #     barrier_h1 = np.sum([np.log(-self.problem.h_1(x, y, i)) for i in range(self.problem.num_constraints_h1)])
    #     barrier_h2 = np.sum([np.log(-self.problem.h_2(x, y, i)) for i in range(self.problem.num_constraints_h2)])
    #     return self.problem.g(x, y) - t * (barrier_h1 + barrier_h2)
    
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
        # grad_tilde_g_y = grad_g_y - t * sum_grad_hy_over_hi
        h_3 = self.problem.h_3(x, y)
        grad_tilde_g_y = grad_g_y - t * sum_grad_hy_over_hi - t * self.problem.gradient_h_3_y(x, y) / h_3
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
        # for i in range(self.problem.num_constraints_h2):
        #     hi = self.problem.h_2(x, y, i)
        #     grad_hi_y = self.problem.gradient_h_2_y(x, y, i)
        #     hessian_hi_yy = self.problem.hessian_h_2_yy(x, y, i)
        #     term = (hessian_hi_yy / hi) - (np.outer(grad_hi_y, grad_hi_y) / hi**2)
        #     sum_constraints += term
        # hessian_tilde_g_yy = hessian_g_yy - t * sum_constraints
        h3 = self.problem.h_3(x, y)
        grad_h3_y = self.problem.gradient_h_3_y(x, y)
        hessian_h3_yy = self.problem.hessian_h_3_yy(x, y)
        term2 = (hessian_h3_yy / h3) - (np.outer(grad_h3_y, grad_h3_y) / h3**2)
        hessian_tilde_g_yy = hessian_g_yy - t * sum_constraints - t * term2
        return hessian_tilde_g_yy
    
    def print_g_expression(self, x, y):
        g_expr = self.tilde_g(x, y)
        print("Expression for g(x, y):")
        print(g_expr)

    def solve_constrained_g(self, x):
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
    
    # def solve_unconstrained_tilde_g(self, x):
    #     n = self.problem.n
    #     y = cp.Variable(n)

    #     tilde_g = self.tilde_g(x, y)
    #     objective = cp.Minimize(tilde_g)

    #     prob = cp.Problem(objective)
    #     try:
    #         prob.solve(solver=cp.ECOS, verbose=False)
    #         if prob.status not in ["optimal", "optimal_inaccurate"]:
    #             print(f"Warning: Optimization problem status {prob.status}")
    #     except cp.SolverError as e:
    #         print("Solver failed:", e)
    #         return y.value

    #     return y.value
    
    def solve_with_interior_point(self, x):
        n = self.problem.n
        y = cp.Variable(n)
        
        objective = cp.Minimize(self.problem.tilde_g(x, y))
        
        constraints = []
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= 0)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= 0)
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, verbose=True)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return None
        
        return y.value

    def solve_constrained_tilde_g(self, x):
        n = self.problem.n
        y = cp.Variable(n)
        
        objective = cp.Minimize(self.tilde_g(x, y))
        
        constraints = []
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= 0)
        # for i in range(self.problem.num_constraints_h2):
        #     constraints.append(self.problem.h_2(x, y, i) <= 0)
        constraints.append(self.problem.h_3(x, y) <= 0)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization problem status {prob.status}")
        except cp.SolverError as e:
            print("Solver failed:", e)
            return y.value
        
        return y.value

    # def inner_loop(self, x, y_init): # Use solver

    #     y_opt = self.solve_unconstrained_tilde_g(x)
    #     grad_y = self.gradient_tilde_g_y(x, y_opt)
    #     grad_norm_y = np.linalg.norm(grad_y)
    #     print(f"gradient norm of g={grad_norm_y}")
    #     return y_opt

    def Interior_inner_loop(self, x, y_init): # Interior point method

        y_opt = self.solve_constrained_g(x)
        y_projected = self.project_to_constraints(x, y_opt)
        grad_y = self.gradient_tilde_g_y(x, y_projected)
        grad_norm_y = np.linalg.norm(grad_y)
        print(f"gradient norm of g={grad_norm_y}")
        return y_projected

    def inner_loop(self, x, y_init): # Accelerated PGD method
        y = y_init.copy()
        y = self.project_to_constraints(x, y)
        for iter in range(self.inner_max_iters):
            if iter == 0:
                w = y
            else:
                w = y + self.beta_y * (y - y_old)
            y_old = y
            grad_w = self.gradient_tilde_g_y(x, w)
            y_new = y - self.alpha_y * grad_w
            # y_projected = self.project_to_constraints(x, y_new)
            grad_y = self.gradient_tilde_g_y(x, y_new)
            grad_norm_y = np.linalg.norm(grad_y)
            print(f"Iterations={iter}, gradient norm={grad_norm_y}")
            if np.linalg.norm(grad_norm_y) < self.epsilon_y:
                print("Inner loop converged at iteration", iter)
                y = y_new
                break
            y = y_new
        return y

    # def inner_loop(self, x, y_init): # Standard PGD method
    #     y = y_init.copy()
    #     for iter in range(self.inner_max_iters):
    #         grad_y = self.gradient_tilde_g_y(x, y)
    #         grad_norm_y = np.linalg.norm(grad_y)
    #         y_new = y - self.alpha_y * grad_y
    #         y_projected = self.project_to_constraints(x, y_new)
    #         print(f"Iterations={iter}, gradient norm={grad_norm_y}")
    #         if np.linalg.norm(grad_norm_y) < self.epsilon_y:
    #             print("Inner loop converged at iteration", iter)
    #             y = y_projected
    #             break
    #         y = y_projected
    #     return y
    
    def upper_loop(self, x_init, y_init):
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

            x_new = x - self.alpha_x * grad_F_x
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
            print(f"f(x, y) = {f_value}, grad_norm of hyperfunction= {np.linalg.norm(grad_F_x)}")
        return x, y, history
