import numpy as np

def f(x,y):
    term1=np.exp(-y+2)/(2+np.cos(6*x))
    term2=0.5*np.log((4*x-2)**2+1)
    return term1+term2

def gradient_f(x,y):
    term1=np.exp(-y+2)/(2+np.cos(6*x))
    term1_x=-term1*(6*np.sin(6*x))/(2+np.cos(6*x))
    term1_y=-term1

    term2=0.5*np.log((4*x-2)**2+1)
    term2_x=4*(4*x-2)/(4*(4*x-2)**2+4)
    term2_y=0

    df_dx=term1_x+term2_x
    df_dy=term1_y+term2_y

    return np.array([df_dx,df_dy])

def g(x,y,t):
    return (y-2*x)**2-t*np.log(x-y)

def gradient_g(x,y,t):
    dg_dx=-4*(y-2*x)-t/(x-y)
    dg_dy=2*(y-2*x)+t/(x-y)
    return np.array([dg_dx,dg_dy])

def hessian_y_g(x,y,t):
    d2g_dy2=2+t/(x-y)**2
    return d2g_dy2

def mixed_hessian_g(x,y,t):
    d2g_dxdy=-4-t/(x-y)**2
    return d2g_dxdy

def project_to_region(x,y,M):
    if y-x<=-M:
        return y
    else:
        return x-M

def projected_gradient_descent(x,t,y_0,M,K,epsilon,alpha):
    y=y_0
    for _ in range(K):
        print("x:",x)
        grad=gradient_g(x,y,t)
        grad_y=grad[1]
        print(grad_y)
        y-=alpha*grad_y
        print("y:",y)
        y=project_to_region(x,y,M)
        print("y:",y)
        grad_norm=np.abs(grad[1])
        if grad_norm<epsilon:
            return float(y), True
    grad=gradient_g(x,y,t)
    grad_norm=np.abs(grad[1])
    return float(y),grad_norm<epsilon

def LL_solver(x,t,y_0,M,K,epsilon,alpha):
    while True:
        y,converged=projected_gradient_descent(x,t,y_0,M,K,epsilon,alpha)
        if converged:
            return float(y)
        else:
            M/=2
            y_0=y

def barrier_method(x_0,y_0,t,I,M,K,epsilon,alpha_1,alpha_2):
    x=x_0
    y=y_0
    for _ in range(I):
        y=LL_solver(x,t,y,M,K,epsilon,alpha_1)
        grad_f=gradient_f(x,y)
        grad_f_x=grad_f[0]
        grad_f_y=grad_f[1]
        hessian_y=hessian_y_g(x,y,t)
        mixed_hessian=mixed_hessian_g(x,y,t)
        descent_direction=grad_f_x-mixed_hessian/hessian_y*grad_f_y
        x=x-alpha_2*descent_direction
        x=np.clip(x,0,3)
    return x,y

x_0=2.0
y_0=1.0
t=0.1
I=100
M=1
K=1000
epsilon=1e-2
alpha_1=0.01
alpha_2=0.05

x_final,y_final=barrier_method(x_0,y_0,t,I,M,K,epsilon,alpha_1,alpha_2)
print("Final x:",x_final)
print("Final y:",y_final)
print("Final function value",f(x_final,y_final))
