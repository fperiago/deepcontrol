# heat_controlñ_nd.py solves the controllability problem
# for the linear heat equation
# y_t = \Delta y in (0,1)^d x (0,T)
# y(x, 0) = norm(x)^2/d
# y(x, t) = u(x,t)      on the whole boundary
# y(x, 1) = norm(x)^2/d + 2


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde
from deepxde.backend import tf

dim = 1

def main():
    def pde(x, y): # heat equation
        dy_t = dde.grad.jacobian(y, x, i=0, j=dim)
        laplacian = 0
        for k in range(dim):
            laplacian = laplacian + dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - laplacian

    def initial_temp(x): # initial temperature
        ini_tem = [np.linalg.norm(x[0, 0:dim]) ** 2]
        for k in range(1, len(x)):
            ini_tem = np.append(ini_tem, np.linalg.norm(x[k, 0:dim]) ** 2)
        return ini_tem.reshape(len(x), 1) / dim 

    def exact_solution(x): # exact solution
        ex_sol = [np.linalg.norm(x[0, 0:dim]) ** 2 / dim + 2 * x[0, dim:(dim+1)]]
        for k in range(1, len(x)):
            ex_sol = np.append(ex_sol, np.linalg.norm(x[k, 0:dim]) ** 2 / dim + 2 * x[k, dim:(dim+1)])
        return ex_sol.reshape(len(x), 1)
    
    def final_temp(x): # final temperature
        fin_tem = [np.linalg.norm(x[0, 0:dim]) ** 2 / dim + 2]
        for k in range(1, len(x)):
            fin_tem = np.append(fin_tem, np.linalg.norm(x[k, 0:dim]) ** 2 / dim + 2)
        return fin_tem.reshape(len(x), 1) 
       
    def boundary_time0(x, on_boundary): # time t = 0
        return on_boundary and np.isclose(x[dim],0)
    
    def boundary_timeT(x, on_boundary): # time t = T
        return on_boundary and np.isclose(x[dim], 1)

    x_min = [0]
    x_min *= dim+1
    x_max = [1]
    x_max *= dim+1 

    geom = dde.geometry.geometry_nd.Hypercube(xmin=x_min, xmax=x_max)
    
    b_time0 = dde.DirichletBC(geom, initial_temp, boundary_time0)
    b_timeT = dde.DirichletBC(geom, final_temp, boundary_timeT) 
    
    data = dde.data.PDE(
        geom, 
        pde, 
        [b_time0, b_timeT], 
        num_domain=20000, 
        num_boundary=3000
        )

    net = dde.maps.FNN([dim+1] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    model.train(epochs=20000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    #Post-processing: error analysis 
     
    X = geom.random_points(1000)
    y_true = exact_solution(X)
    y_pred = model.predict(X)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))       
                                                                           
                                       
if __name__ == "__main__":
    main()
