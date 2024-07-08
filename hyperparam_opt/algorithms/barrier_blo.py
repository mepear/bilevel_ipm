import cvxpy as cp
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

import sys
sys.path.append('..')

from utils import load_diabetes, train_val_test_split

class Barrier_BLO:
    """
    Write our barrier BLO method into a class
    """
    def __init__(self, problem, hparams, epochs=100, verbose=True):
        self.problem = problem
        self.hparams = hparams
        
        self.t = hparams['t']
        self.epochs = epochs
        self.verbose = verbose
    
    # def projected_gradient_descent(self, x,t,y_0,M,max_iters_inner,epsilon_1,epsilon_2,alpha):
    def projected_gradient_descent(self, c0, w0, b0, xi0, M, max_iters_inner, epsilon, alpha):
        c = c0
        # print(f"before projection c: {c0}, w: {w0}, b: {b0}, xi: {xi0}")
        if not self.problem.check_constraints(c0, w0, b0, xi0, m=M):
            w, b, xi = self.problem.proj_to_lower_constraints(c0, w0, b0, xi0, m=M)
        # print(f"after projection c: {c}, w: {w}, b: {b}, xi: {xi}")
        # print(f"constraints: {self.problem.lower_constraints(c, w, b, xi, m=0.0)}")
        i, grad_norm = 0, float('inf')
        while grad_norm > epsilon and i < max_iters_inner:
            grad_w, grad_b, grad_xi = self.problem.lower_grad_y(c, w, b, xi)
            w_old, b_old, xi_old = w, b, xi
            # print(grad_w, grad_b, grad_xi)
            # print(grad_w.shape, grad_b.shape, grad_xi.shape)
            K = max(M, 1e-2)
            w, b, xi = w_old - K * alpha * grad_w, b_old - K * alpha * grad_b, xi_old - K * alpha * grad_xi
            # print(f"    Inner loop PGD total iter: {i}, w old: {w}, b old: {b}, xi old: {xi}")
            if not self.problem.check_constraints(c0, w0, b0, xi0, m=M):
                w, b, xi = self.problem.proj_to_lower_constraints(c, w, b, xi, m=M)
            # print(f"    Inner loop PGD total iter: {i}, w new: {w}, b new: {b}, xi new: {xi}")
            # delta = np.linalg.norm(w_old - w) + np.linalg.norm(b_old - b) + np.linalg.norm(xi_old - xi)
            # if delta < 0.005:
            #    break
            # delta_w, delta_b, delta_xi = w - w_old, b - b_old, xi - xi_old
            grad_norm = np.linalg.norm(grad_w) + np.linalg.norm(grad_b) + np.linalg.norm(grad_xi)
            # print(f"    Inner loop PGD total iter: {i}, gradient w: {grad_w}, gradient b: {grad_b}, gradient xi: {grad_xi}")
            # print(f"    Inner loop PGD curr iter: {i}, grad norm: {grad_norm}, M: {M}")
            i += 1
            
        grad_w, grad_b, grad_xi = self.problem.lower_grad_y(c, w, b, xi)
        grad_norm = np.linalg.norm(grad_w) + np.linalg.norm(grad_b) + np.linalg.norm(grad_xi)
        print(f"    Inner loop PGD total iter: {i}, grad norm: {grad_norm}, M: {M}")
        return w, b, xi, grad_norm < epsilon
    
    def lower_loop(self, c0, w0, b0, xi0, M, max_iters_inner, epsilon, alpha, lower_bound_M=1e-6):
        converged = False
        while not converged and M >= lower_bound_M:
            w, b, xi, converged=self.projected_gradient_descent(c0, w0, b0, xi0, M, max_iters_inner, epsilon, alpha)
            M /= 2
            w0, b0, xi0 = w, b, xi
            # break # this break is just for debugging
        return w, b, xi, M * 2
    
    def upper_loop(self, c0, w0, b0, xi0, hparams):
        """
        alpha, beta: corresponds to lower and upper lr
        """
        M, max_iters_outer, max_iters_inner, epsilon, alpha, beta =\
            hparams['M'], hparams['max_iters_outer'], hparams['max_iters_inner'], hparams['epsilon'], hparams['alpha'], hparams['beta']

        val_loss_list=[]
        test_loss_list=[]
        val_acc_list=[]
        test_acc_list=[]
        time_computation=[]
        metrics = []
        algorithm_start_time=time.time()
        
        # variables = []
        
        c = c0
        w, b, xi = w0, b0, xi0
        curr_metric = self.problem.compute_metrics(c, w, b, xi)
        curr_metric['time_computation'] = time.time() - algorithm_start_time
        metrics.append(curr_metric)
        epoch = 0
        grad_norm = float('inf')
        while grad_norm > epsilon and epoch < max_iters_outer:
            # variables.append({'c': c, 'xi': xi, 'w': w, 'b': b}) # uncomment this if you want to see all variables
            w, b, xi, M = self.lower_loop(c, w, b, xi, 0.5, max_iters_inner, epsilon, alpha)
            grad_c = self.problem.upper_grad_x(c)
            grad_w, grad_b, grad_xi = self.problem.upper_grad_y(c, w, b, xi)
            hessian_y = self.problem.lower_hessian(c, w, b, xi)
            jacobian = self.problem.lower_jacobian(c, w, b, xi)
            print(f"hessian_y shape: {hessian_y.shape}, jacobian shape: {jacobian.shape}, np.block([grad_w, grad_b, grad_xi]) shape: {np.block([grad_w, grad_b, grad_xi]).shape}")
            descent_direction = grad_c - jacobian.dot(np.linalg.solve(hessian_y, np.block([grad_w, grad_b, grad_xi])))
            c = c - beta * descent_direction
            # c[c <= 1.5] = 1.5

            grad_norm=np.linalg.norm(descent_direction)
            print(f"Upper iter: {epoch}, grad norm of hyperfunction: {grad_norm}")
            
            # update metrics
            curr_metric = self.problem.compute_metrics(c, w, b, xi)
            curr_metric['time_computation'] = time.time() - algorithm_start_time
            metrics.append(curr_metric)
            
            if epoch%5==0 and self.verbose:
                # print(f"c: {c}, w: {w}, b: {b}, xi: {xi}")
                print(f"Epoch [{epoch}/{max_iters_outer}]:",
                "val acc: {:.2f}".format(curr_metric['val_acc']),
                "val loss: {:.2f}".format(curr_metric['val_loss']),
                "test acc: {:.2f}".format(curr_metric['test_acc']),
                "test loss: {:.2f}".format(curr_metric['test_loss']))
                # print(f"Epoch [{j}/{epoch}]:","upper_loss: ", loss_upper.detach().numpy()/15.0, "test_loss_upper: ", test_loss_upper.detach().numpy()/11.8)

            val_loss_list.append(curr_metric['val_loss']) # length 150
            test_loss_list.append(curr_metric['test_acc']) # length 118
            val_acc_list.append(curr_metric['val_acc'])
            test_acc_list.append(curr_metric['test_acc'])
            curr_metric['time_computation'] = time.time() - algorithm_start_time
            time_computation.append(curr_metric['time_computation'])
            epoch += 1
            
        print(f"Outer loop total iter: {epoch}, final grad norm: {grad_norm}")
        return metrics, c, w, b, xi, grad_norm
    
class SVM_Problem:
    """
    Define the SVM problem into a class
    in this class, variables c, w, b and xi are numpy arrays
    """
    def __init__(self, datasets, t=1e-3):
        self.x_train = datasets["x_train"]
        self.y_train = datasets["y_train"]
        self.x_val = datasets["x_val"]
        self.y_val = datasets["y_val"]
        self.x_test = datasets["x_test"]
        self.y_test = datasets["y_test"]
        
        self.feature = self.x_train.shape[1]
        self.t = t
    
    def f_val(self, c, w, b, xi):
        """
        Upper objective
        Right now this function is written in torch but we should d
        """
        # try:
        #     w_tensor = torch.Tensor(w) #.requires_grad_()
        # except:
        #     print(w_tensor)
        #     raise RuntimeError("HE DADO NONE")
        # b_tensor = torch.Tensor(np.array([b.value])) #.requires_grad_()
        # xi_tensor = torch.Tensor(np.array([xi.value]))
        # c_tensor = torch.Tensor(np.array([c.value])).requires_grad_()
        
        # x = torch.reshape(torch.Tensor(self.y_val), (torch.Tensor(self.y_val).shape[0],1)) 
        # x = x * F.linear(torch.Tensor(self.x_val), w_tensor, b_tensor) # / torch.linalg.norm(w_tensor)
        # loss_upper = torch.sum(torch.exp(1-x)) + torch.linalg.norm(c_tensor)  # TODO: isn't this norm square???
        
        x = self.y_val.reshape((self.y_val.shape[0], 1))
        x = x.dot(self.x_val.dot(w) + b)
        loss_upper = np.sum(np.exp(1 - x)) + 0.5 * np.linalg.norm(c)**2
        
        return loss_upper 
    
    def g_val(self, w):
        """lower objective"""
        return 0.5*cp.norm(w, 2)**2
    
    def upper_constraints(self):
        return []
    
    def proj_to_upper_constraints(self, c):
        return c
        
    def lower_constraints(self, c, w, b, xi, m=0.0):
        constraints=[]
        for i in range(self.y_train.shape[0]):
            # print(f"xi[i]: {xi[i]}, self.y_train[i]: {self.y_train[i]}, self.x_train[i]: {self.x_train[i]}, b: {b}")
            constraints.append(1 - xi[i] - self.y_train[i] * (self.x_train[i] @ w + b) <= -m)
        
        constraints.extend([xi - c <= - m])
        # print(f"[xi <= c - m]: {[xi - c <= - m]}")
        return constraints
    
    def check_constraints(self, c0, w0, b0, xi0, m):
        # Check the constraints for each sample
        for i in range(len(self.y_train)):
            if not np.all(1 - self.x_train[i] - self.y_train[i] * (self.x_train[i] @ w0 + b0) <= -m):
                return False
            
        # Check the additional constraints on xi
            if not all(xi0 - c0 <= -m):
                return False
        return True


    def proj_to_lower_constraints(self, c0, w0, b0, xi0, m):
        """Using cvxpy to do the projection"""
        d = self.feature
        # # upper variable
        # c = cp.Parameter(y_train.shape[0], nonneg=True)
        
        # lower variables
        w = cp.Variable(d)
        b = cp.Variable()
        xi = cp.Variable(self.y_train.shape[0], nonneg=True)
 
        # setup the objective and constraints and solve the problem
        obj = cp.Minimize(cp.sum_squares(w - w0) + cp.sum_squares(b - b0) + cp.sum_squares(xi - xi0))
        constr = self.lower_constraints(c0, w, b, xi, m=m)
        
        # ccc = self.lower_constraints(c0, w0, b0, xi0, m=m)
        # print(f"constraint before solving: {ccc}")
        prob = cp.Problem(obj, constr)
        try:
            prob.solve()
            # print(f"constraint after solving: {[v.value for v in prob.constraints]}")
        except:
            print(prob.status)
            raise RuntimeError("The projection problem is not solvable")
    
        return np.array(w.value), np.array(b.value), np.array(xi.value)

    def tilde_g_val(self, c, w, b, xi, m=0.0):
        return self.g_val(w) + sum([-self.t * math.log(-v) for v in self.lower_constraints(c, w, b, xi, m=m)])
    
    def upper_grad_x(self, c):
        return c
    
    def lower_grad_x(self, c, w, b, xi, t=1e-3):
        return - self.y_train.shape[0] * self.t * sum([1 / (c[i] - xi[i]) for i in range(self.y_train.shape[0])])
    
    def upper_grad_y(self, c, w, b, xi):
        grad_w = np.zeros(self.feature)
        grad_b = 0.0
        grad_xi = np.zeros(xi.shape)
        
        for i in range(self.y_val.shape[0]):
            temp = np.exp(1 - self.y_val[i] * (self.x_train[i].dot(w) + b))
            grad_w -= temp * self.y_val[i] * self.x_val[i]
            grad_b -= temp * self.y_val[i]
        
        return grad_w, grad_b, grad_xi
    
    def lower_grad_y(self, c, w, b, xi):
        grad_w = np.array(w)
        grad_b = np.array(0.0)
        # print(c.shape, w.shape, b.shape, xi.shape, self.x_train[0].shape, self.y_train[0].shape)
        # print(f"w: {w}, b: {b}, c: {c}")
        # print(f"self.y_train: {self.y_train}")
        
        for i in range(self.y_train.shape[0]):
            temp = 1 / (self.y_train[i] * (self.x_train[i].dot(w) + b) + xi[i] - 1)
            # print(f"self.x_train[i].dot(w) + b + xi[i] - 1: {self.x_train[i].dot(w) + b + xi[i] - 1}")
            # print(f"temp: {temp}")
            grad_w -= self.t * temp * self.y_train[i] * self.x_train[i]
            grad_b -= self.t * temp * self.y_train[i]
        
        grad_xi = np.array([-self.t * temp + self.t * 1 / (c[i] - xi[i]) for i in range(self.y_train.shape[0])])
        
        return grad_w, grad_b, grad_xi
    
    def lower_hessian(self, c, w, b, xi):
        """This is a HUGE matrix"""
        h11 = np.eye(self.feature) # nabla_w^2
        h12 = np.zeros(shape=(self.feature, 1)) # nabla_w nabla_b
        h13 = np.zeros(shape=(self.feature, self.y_train.shape[0])) # nabla_w nabla_xi
        h22 = np.zeros(1) # nabla_b^2
        h23 = np.zeros(shape=(1, self.y_train.shape[0])) # nabla_b nabla_xi
        h33 = np.zeros(shape=(self.y_train.shape[0], self.y_train.shape[0])) # nabla_xi^2
        
        for i in range(self.y_train.shape[0]):
            temp = 1 / (self.y_train[i] * (self.x_train[i].dot(w) + b) + xi[i] - 1)**2
            # print(f"temp: {temp}, self.y_train[i]: {self.y_train[i]}, self.x_train[i] shape: {self.x_train[i].shape}")
            h11 += self.t * temp * self.y_train[i]**2 * self.x_train[i].reshape((self.feature, 1)).dot(self.x_train[i].reshape((1, self.feature)))
            h12 += self.t * temp * self.y_train[i]**2 * self.x_train[i].reshape((self.feature, 1))
            h13[:, i] = self.t * temp * self.y_train[i] * self.x_train[i] # .reshape((self.feature, 1))
            h22 += self.t * temp * self.y_train[i]**2
            h23[0, i] = self.t * temp * self.y_train[i]
            h33[i, i] = self.t * temp + self.t / (c[i] - xi[i])**2
        
        # print(f"shapes: h11: {h11.shape}, h12: {h12.shape}, h13: {h13.shape}, h22: {h22.shape}, h23: {h23.shape}, h33: {h33.shape}")
        return np.block(
            [
                [h11, h12, h13],
                [h12.T, h22, h23],
                [h13.T, h23.T, h33]
            ]
        )
    
    def lower_jacobian(self, c, w, b, xi):
        """nabla_x nabla_y tilde_g"""
        return np.block([np.zeros((self.y_train.shape[0], self.feature)), np.zeros((self.y_train.shape[0], 1)), -self.t * np.diag([1 / (c[i] - xi[i])**2 for i in range(self.y_train.shape[0])])])
    
    def inv_hessian(self,  c, w, b, xi):
        """inverse of hessian"""
        return np.linalg.inv(self.lower_hessian(c, w, b, xi))
    
    def approximate_inv_hessian(self, h=10):
        """approximate inverse of hessian using Neumann series"""
        pass
    
    def compute_metrics(self, c, w, b, xi):
        x = self.y_val # .reshape((self.y_val.shape[0], 1))
        x = np.multiply(x, self.x_val.dot(w) + b)
        # loss_upper = np.sum(np.exp(1 - x)) + 0.5 * np.linalg.norm(c)**2
        
        x1 = self.y_test # .reshape((self.y_test.shape[0], 1))
        x1 = np.multiply(x1, self.x_test.dot(w) + b)
        test_loss_upper = np.sum(np.exp(1 - x1))
        
        val_loss = np.sum(np.exp(1 - x)) / self.y_val.shape[0]
        test_loss = test_loss_upper / self.y_test.shape[0]

        ###### Accuracy
        q = self.y_train # .reshape((self.y_train.shape[0], 1))
        q = np.multiply(q, self.x_train.dot(w) + b)
        # print(f"self.y_train shape: {self.y_train.shape}, self.x_train shape: {self.x_train.shape}, w and b shape: {w.shape, b.shape}, self.x_train.dot(w) + b: {self.x_train.dot(w) + b}, q: {q}")
        train_acc = (q > 0).sum() / len(self.y_train)

        q = self.y_val # .reshape((self.y_val.shape[0], 1))
        q = np.multiply(q, self.x_val.dot(w) + b)
        val_acc = (q > 0).sum() / len(self.y_val)

        q = self.y_test # .reshape((self.y_test.shape[0], 1))
        q = np.multiply(q, self.x_test.dot(w) + b)
        test_acc = (q > 0).sum() / len(self.y_test)
        
        return {
            #'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            # 'loss_upper': loss_upper,
            # 'loss_lower': loss_lower,
            # 'time_computation': time.time() - algorithm_start_time
        }


if __name__ == "__main__":
    ############ Load data code ###########

    data_utils = load_diabetes()

    data_list=[]

    f = open("../diabete.txt",encoding = "utf-8")
    a_list=f.readlines()
    f.close()
    for line in a_list:
        line1=line.replace('\n', '')
        line2=list(line1.split(' '))
        y=float(line2[0])
        x= [float(line2[i].split(':')[1]) for i in (1,2,3,4,5,6,7,8)]
        data_list.append(x+[y])


    data_array_1=np.array(data_list)[:,:-1]
    data_array_0=np.ones((data_array_1.shape[0],1))
    data_array_2=data_array_1*data_array_1
    data_array_3=np.empty((data_array_1.shape[0],0))

    for i in range(data_array_1.shape[1]):
        for j in range(data_array_1.shape[1]):
            if i<j:
                data_array_i=data_array_1[:,i]*data_array_1[:,j]
                data_array_i=np.reshape(data_array_i,(-1,1))
                data_array_3=np.hstack((data_array_3,data_array_i))

    data_array_4=np.reshape(np.array(data_list)[:,-1],(-1,1))
    data=np.hstack((data_array_0,data_array_1,data_array_2,data_array_3,data_array_4))

    n_train = 500
    n_val = 150

    metrics = []
    variables = []

    hparams = {
        'gam': 5,
        # 'eta': 0.1,
        'alpha': 0.01,
        'beta': 0.01,
        'epsilon': 1e-5,
        'max_iters_outer': 100,
        'max_iters_inner': 100,
        'M': 0.05,
        't': 1e-3
    }

    epochs = 80
    plot_results = True

    c0, w0, b0, xi0 = 10 * np.ones(n_train), np.random.randn(45), np.random.randn(), np.random.randn(n_train)
    for seed in range(10):

        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, seed, n_train, n_val)
        datasets = {
            'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 
            'y_val': y_val, 'x_test': x_test, 'y_test': y_test
        }
        
        problem = SVM_Problem(datasets)
        barrier_blo = Barrier_BLO(problem, hparams)
        
        # metrics_seed, variables_seed = barrier_blo(x_train, y_train, x_val, y_val, x_test, y_test, hparams, epochs)
        metrics_seed, c, w, b, xi, grad_norm = barrier_blo.upper_loop(c0, w0, b0, xi0, hparams)
        metrics.append(metrics_seed)
        # variables.append([c, w, b, xi])

    train_acc = np.array([[x['train_acc'] for x in metrics] for metrics in metrics])
    val_acc = np.array([[x['val_acc'] for x in metrics] for metrics in metrics])
    test_acc = np.array([[x['test_acc'] for x in metrics] for metrics in metrics])

    val_loss = np.array([[x['val_loss'] for x in metrics] for metrics in metrics])
    test_loss = np.array([[x['test_loss'] for x in metrics] for metrics in metrics])

    time_computation = np.array([[x['time_computation'] for x in metrics] for metrics in metrics])

    if plot_results:
        val_loss_mean=np.mean(val_loss,axis=0)
        val_loss_sd=np.std(val_loss,axis=0)/2.0
        test_loss_mean=np.mean(test_loss,axis=0)
        test_loss_sd=np.std(test_loss,axis=0)/2.0

        val_acc_mean=np.mean(val_acc,axis=0)
        val_acc_sd=np.std(val_acc,axis=0)/2.0
        test_acc_mean=np.mean(test_acc,axis=0)
        test_acc_sd=np.std(test_acc,axis=0)/2.0

        axis = np.mean(time_computation,axis=0)

        plt.rcParams.update({'font.size': 18})
        plt.rcParams['font.sans-serif']=['Arial']
        plt.rcParams['axes.unicode_minus']=False
        axis=time_computation.mean(0)
        plt.figure(figsize=(8,6))
        #plt.grid(linestyle = "--")
        ax = plt.gca()
        plt.plot(axis,val_loss_mean,'-',label="Training loss")
        ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
        plt.plot(axis,test_loss_mean,'--',label="Test loss")
        ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        #plt.legend(loc=4)
        plt.ylabel("Loss")
        #plt.xlim(-0.5,3.5)
        #plt.ylim(0.5,1.0)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold')
        plt.savefig('ho_svm_kernel_1.pdf') 
        #plt.show()

        plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.plot(axis,val_acc_mean,'-',label="Training accuracy")
        ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
        plt.plot(axis,test_acc_mean,'--',label="Test accuracy")
        ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
        #plt.xticks(np.arange(0,iterations,40))
        plt.title('Kernelized SVM')
        plt.xlabel('Running time /s')
        plt.ylabel("Accuracy")
        # plt.ylim(0.64,0.8)
        #plt.legend(loc=4)
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        #plt.setp(ltext, fontsize=18,fontweight='bold')
        plt.savefig('ho_svm_kernel_2.pdf') 
        plt.show()
