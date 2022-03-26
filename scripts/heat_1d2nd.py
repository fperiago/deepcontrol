# heat_control_1d.py solves the boundary  controllability problem
# for the linear heat equation
# y_t = y_xx      in (0,1)x(0,T)
# y(x,0)=x^2
# y(0,t)=u_1(t), y(1,t)=u_2(t)  
# y(x,1)=x^2 + 1


import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde


import pathlib
import os

OUTPUT_DIRECTORY = pathlib.Path.cwd() / "results" / "heat_1d2nd"
if not OUTPUT_DIRECTORY.exists():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def pde(x, y): # heat equation
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    return dy_t -  dy_xx

def initial_temp(x): # initial temperature

    return x[:, 0:1]**2    

def final_temp(x):

    return x[:, 0:1]**2 + 2 

def exact_solution(x): # exact solution

    return x[:, 0:1]**2 + 2 * x[:,1:2]
   
def boundary_bottom(x, on_boundary): # boundary t =0

    return (on_boundary and np.isclose(x[1],0))

def boundary_upper(x, on_boundary): # boundary t = 1
    return (on_boundary and np.isclose(x[1], 1))

geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

b_time0 = dde.DirichletBC(geom, initial_temp, boundary_bottom)
b_timeT = dde.DirichletBC(geom, final_temp, boundary_upper) 

data = dde.data.PDE(
    geom, 
    pde, 
    [b_time0, b_timeT], 
    num_domain=1000, 
    num_boundary=300,
    solution=exact_solution,
    num_test=100
    )

net = dde.maps.FNN([2] + [40] * 4 + [1], "tanh", "Glorot uniform")

model = dde.Model(data, net)

model.compile("adam", lr=0.001)

model.train(epochs=20000)

model.compile("L-BFGS-B")

losshistory, train_state = model.train()

dde.saveplot(
    losshistory, train_state, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)

#Post-processing: error analysis and figures 
 
X = geom.random_points(1000)
y_true = exact_solution(X)
y_pred = model.predict(X)

print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))   

XX, T = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100)
)

def explicit_solution(x,t): # exact solution
    return x**2 + 2 * t

Y = explicit_solution(XX, T)            

fig = plt.figure() # plot of exact state
ax = plt.axes(projection="3d")
surf = ax.plot_surface(XX, T, Y)

ax.set_title('exact state ')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')

plt.show()
    
