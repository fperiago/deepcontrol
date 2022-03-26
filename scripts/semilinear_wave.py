# semilinear_wave.py solves the boundary  controllability problem
# for the semilinear wave equation
# y_tt-y_xx = 4y^2 in (0,1) x (0,2)
# y(x,0) = 1.5 *s in(3*pi*x), y_t(x,0) = x^2
# y(0,t) = 0, y(1,t) = u(t)
# y(x, 2) = y_t(x,2) = 0


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import deepxde as dde


import pathlib
import os

OUTPUT_DIRECTORY = pathlib.Path.cwd() / "results" / "semilinear_wave"
if not OUTPUT_DIRECTORY.exists():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

contador = 0
contador2 = 0


def pde(x, y):  # semilinear wave equation
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    return dy_tt - dy_xx - 4 * y * y


def initial_pos(x):  # initial position

    return 1.5 * np.sin(np.pi * 3 * x[:, 0:1])


def initial_velo(x):  # initial velocity

    return -x * x  # minus sign is because the exterior normal vector is n=(0,-1)


def boundary_left(x, on_boundary):  # boundary x = 0

    return on_boundary and np.isclose(x[0], 0)


def boundary_bottom(x, on_boundary):  # boundary t = 0
    is_on_boundary_bottom = (
        on_boundary
        and np.isclose(x[1], 0)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], 1)
    )

    return is_on_boundary_bottom


def boundary_upper(x, on_boundary):  # boundary t = T
    is_on_boundary_upper = (
        on_boundary
        and np.isclose(x[1], 2)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], 1)
    )

    return is_on_boundary_upper


geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2])

bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_left)
bc2 = dde.DirichletBC(geom, initial_pos, boundary_bottom)
bc3 = dde.NeumannBC(geom, initial_velo, boundary_bottom)
bc4 = dde.DirichletBC(geom, lambda x: 0, boundary_upper)
bc5 = dde.NeumannBC(geom, lambda x: 0, boundary_upper)


data = dde.data.PDE(
    geom, pde, [bc1, bc2, bc3, bc4, bc5], num_domain=5625, num_boundary=225
)

net = dde.maps.FNN([2] + [100] * 5 + [1], "tanh", "Glorot uniform")

model = dde.Model(data, net)

model.compile("adam", lr=0.001)

model.train(epochs=20000)

model.compile("L-BFGS-B")

losshistory, train_state = model.train()

dde.saveplot(
    losshistory, train_state, issave=True, isplot=True, output_dir=OUTPUT_DIRECTORY
)


# Post-processing: error analysis and figures

tt = np.linspace(0, 2, 100)
xx = np.ones_like(tt)
X = np.vstack((xx, tt)).T

control_predict = np.ravel(model.predict(X))  # predicted control
control_predict[0] = 0  # at t=0 this value is known

fig, ax = plt.subplots()
ax.plot(tt, control_predict, "b", linewidth=2, label="Predicted control")
ax.set_xlabel("$t$")
ax.grid()

plt.show()

X = np.linspace(0, 1, 100)
t = np.linspace(0, 2, 200)

X_repeated = np.repeat(X, t.shape[0])
t_tiled = np.tile(t, X.shape[0])
XX = np.vstack((X_repeated, t_tiled)).T

state_predict = model.predict(XX).T  # predicted state
state_predict_M = state_predict.reshape((100, 200)).T  # predicted state in matrix form

Xx, Tt = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 200))

fig = plt.figure()  # plot of predicted state
ax = plt.axes(projection="3d")
surf = ax.plot_surface(Xx, Tt, state_predict_M)

ax.set_title("PINN state ")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

plt.show()
