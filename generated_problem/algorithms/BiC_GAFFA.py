import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import time

class BiC_GAFFA:
    def __init__(self, problem, hparams):
        self.problem = problem
        bic_gaffa_params = hparams.get('bic_gaffa', {})

        self.alpha = bic_gaffa_params.get('alpha', 1e-4)
        self.c = bic_gaffa_params.get('c', 1e-4)
        self.tau = bic_gaffa_params.get('tau', 1e-4)
        self.max_iters = bic_gaffa_params.get('max_iters', 50)
        self.gamma_1 = bic_gaffa_params.get('gamma_1', 1)
        self.gamma_2 = bic_gaffa_params.get('gamma_2', 1)
        self.eta = bic_gaffa_params.get('eta', 1)
        self.r = bic_gaffa_params.get('r', 1)
        self.epsilon = bic_gaffa_params.get('epsilon', 1)
    
    def compute_inner_product_x(self, x, y, z):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_x = np.zeros((total_constraints, n))
        
        for i in range(num_constraints_h1):
            H_x[i, :] = self.problem.gradient_h_1_x(x, y, i)
        
        for i in range(num_constraints_h2):
            H_x[num_constraints_h1 + i, :] = self.problem.gradient_h_2_x(x, y, i)
        
        z_grad_y = H_x.T @ z
        
        return z_grad_y
    
    def compute_inner_product_y(self, x, y, z):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_y = np.zeros((total_constraints, n))
        
        for i in range(num_constraints_h1):
            H_y[i, :] = self.problem.gradient_h_1_y(x, y, i)
        
        for i in range(num_constraints_h2):
            H_y[num_constraints_h1 + i, :] = self.problem.gradient_h_2_y(x, y, i)
        
        z_grad_y = H_y.T @ z
        
        return z_grad_y
    
    def constraint_vector(self, x, y):
        constraints = []
    
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i))
    
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i))
    
        constraint_vector = np.array(constraints)
        return constraint_vector
    
    def project_to_joint_constraints(self, x_init, y_init):
        n = self.problem.n
        xy_init = np.hstack((x_init, y_init))
    
        def objective(xy):
            x = xy[:n]
            y = xy[n:]
            return np.sum((x - x_init)**2 + (y - y_init)**2)
    
        def constraints_func(xy):
            x = xy[:n]
            y = xy[n:]
            cons = []
            for i in range(self.problem.num_constraints_h1):
                cons.append(self.problem.h_1(x, y, i))
            for i in range(self.problem.num_constraints_h2):
                cons.append(self.problem.h_2(x, y, i))
            return np.array(cons)
    
        cons = {'type': 'ineq', 'fun': constraints_func}
    
        res = minimize(objective, xy_init, method='trust-constr', constraints=cons,
                   options={'maxiter': 1000})
    
        x_proj = res.x[:n]
        y_proj = res.x[n:]
        return x_proj, y_proj

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
    
    def bic_gaffa(self, x_init, y_init, z_init, theta_init):
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        theta = theta_init.copy()
        c = self.c
        history = []
        start_time = time.time()

        for iter in range (self.max_iters):
            inner_product_y_x_theta_z = self.compute_inner_product_y(x, theta, z)
            gradient_g_y_x_theta = self.problem.gradient_g_y(x, theta)
            d_theta = gradient_g_y_x_theta  + inner_product_y_x_theta_z + (1 / self.gamma_1) * (theta - y)

            theta = self.project_to_constraints(x, theta - self.eta * d_theta)

            constraint_vector = self.constraint_vector(x, y)
            lambd = z + self.gamma_2 * constraint_vector
            lambd = np.maximum(lambd, 0)

            gradient_f_x_x_y = self.problem.gradient_f_x(x, y)
            gradient_g_x_x_y = self.problem.gradient_g_x(x, y)
            inner_product_x_x_y_lambd = self.compute_inner_product_x(x, y, lambd)
            gradient_g_x_x_theta = self.problem.gradient_g_x(x, theta)
            inner_product_x_x_theta_z = self.compute_inner_product_x(x, theta, z)
            d_x = (1 / c) * gradient_f_x_x_y + gradient_g_x_x_y + inner_product_x_x_y_lambd - gradient_g_x_x_theta - inner_product_x_x_theta_z

            gradient_f_y_x_y = self.problem.gradient_f_y(x, y)
            gradient_g_y_x_y = self.problem.gradient_g_y(x, y)
            inner_product_y_x_y_lambd = self.compute_inner_product_y(x, y, lambd)
            d_y = (1 / c) * gradient_f_y_x_y + gradient_g_y_x_y + inner_product_y_x_y_lambd - (y - theta) / self.gamma_1
            
            g_x_theta = self.problem.g(x, theta)
            d_z = -(z - lambd)/self.gamma_2 - g_x_theta
            
            print(f"norm of dx, dy, dx: {np.linalg.norm(d_x), np.linalg.norm(d_y), np.linalg.norm(d_z)}")

            x_new = x - self.alpha * d_x
            
            print(f"before: norm of y {np.linalg.norm(y)}")
            y_new = self.project_to_constraints(x, y - self.alpha * d_y)
            print(f"after: norm of y {np.linalg.norm(y_new)}")

            z_new = z - self.alpha * d_z
            z_new = np.maximum(z, 0)
            z_new = np.minimum(z, self.r)

            # if (np.linalg.norm(d_x) <= self.epsilon) and (np.linalg.norm(y_new - y) / self.alpha <= self.epsilon) and (np.linalg.norm(z_new - z) / self.alpha <= self.epsilon):
            #     print("Main loop converged at iteration", iter)
            #     break
            
            c = c * self.tau
            x = x_new
            y = y_new
            z = z_new
            elapsed_time = time.time() - start_time

            f_value = self.problem.f(x, y)
            history.append({
                'iteration': iter,
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': np.linalg.norm(d_x) + np.linalg.norm(y_new - y) / self.alpha + np.linalg.norm(z_new - z) / self.alpha,
                'time': elapsed_time
            })
            print(f"f(x,y)={f_value}")
        print(f"end: norm of y {np.linalg.norm(y)}")
        return x, y, history


if __name__=="__main__":
    import numpy as np
    import cvxpy as cp

    from problems.data_generation import generate_problem_data
    from problems.problem_definition import BilevelProblem
    from algorithms.barrier_blo_box import BarrierBLO
    from algorithms.blocc import BLOCC
    from algorithms.implicit_gradient_descent import IGD
    from algorithms.BiC_GAFFA import BiC_GAFFA
    from algorithms.BSG import BSG

    n = 60
    seed = 9

    data = generate_problem_data(n, seed)
    problem = BilevelProblem(data)

    x_init = np.zeros(n)
    y_init = np.zeros(n)
    z_init = np.zeros(problem.num_constraints_h1 + problem.num_constraints_h2)
    theta_init = np.zeros(n)
    y_g_init = np.zeros(n)
    y_F_init = np.zeros(n)
    mu_g_init = np.zeros(problem.num_constraints_h1 + problem.num_constraints_h2)
    mu_F_init = np.zeros(problem.num_constraints_h1 + problem.num_constraints_h2)
    hparams = {
            'bic_gaffa':{
                'alpha':0.001,
                'c':1,
                'tau':1.3,
                'gamma_1':10,
                'gamma_2':1,
                'eta':1,
                'r':1,
                'epsilon':0.1,
                'max_iters':100
            }
        }   

    bic_algo = BiC_GAFFA(problem, hparams)

    x_opt_bic, y_opt_bic, history = bic_algo.bic_gaffa(x_init, y_init, z_init, theta_init)
    print(problem.f(x_opt_bic, y_opt_bic))