# linear_wave.py solves the boundary  controllability problem
# for the  linear wave equation
#
# y_tt=y_xx in (0,1)x(0,2)
# y(x,0)=sin(pi*x), y_t(x,0)=0, 0<x<1
# y(0,t)=0, y(1,t)=u(t),        0<t<2
# y(x,2)=y_t(x,2)=0,            0<x<1.


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import deepxde as dde

import pathlib
import os

OUTPUT_DIRECTORY = pathlib.Path.cwd() / "results" / "linear_wave"
if not OUTPUT_DIRECTORY.exists():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def pde(x, y):  # wave equation
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    return dy_tt - dy_xx


def initial_pos(x):  # initial position

    return np.sin(np.pi * x[:, 0:1])


def initial_velo(x):  # initial velocity

    return 0.0


def boundary_left(x, on_boundary):  # boundary x=0
    is_on_boundary_left = on_boundary and np.isclose(x[0], 0)

    return is_on_boundary_left


def boundary_bottom(x, on_boundary):  # boundary t=0
    is_on_boundary_bottom = (
        on_boundary
        and np.isclose(x[1], 0)
        and not np.isclose(x[0], 0)
        and not np.isclose(x[0], 1)
    )

    return is_on_boundary_bottom


def boundary_upper(x, on_boundary):  # boundary t=2
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
    geom,
    pde,
    [bc1, bc2, bc3, bc4, bc5],
    num_domain=100,
    num_boundary=30,
)

# training with regularization
# net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform", regularization = ['l2',1e-7])

net = dde.maps.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

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

control_exact = np.piecewise(
    tt,
    [tt <= 1, tt > 1],
    [
        lambda tt: 0.5 * np.sin(np.pi * (1 - tt)),
        lambda tt: -0.5 * np.sin(np.pi * (tt - 1)),
    ],
)  # exact control


gen_error_control = np.linalg.norm(control_predict - control_exact)
print("generalization error = ", gen_error_control)

relative_error_control = np.linalg.norm(
    control_predict - control_exact
) / np.linalg.norm(control_exact)
print("relative error = ", relative_error_control)


fig, ax = plt.subplots()
ax.plot(tt, control_exact, "-b", linewidth=2, label="Exact control")
ax.plot(tt, control_predict, "--r", linewidth=2, label="Predicted control")
ax.set_xlabel("$t$")

plt.show()


def auxiliary(z):
    y_exact = np.piecewise(
        z,
        [(z < -1) | (z > 1), (z >= -1) & (z <= 1)],
        [lambda z: 0.0, lambda z: np.sin(np.pi * z)],
    )
    return y_exact


def explicit_state(x1, t1):
    y_explicit = 0.5 * (auxiliary(x1 - t1) + auxiliary(x1 + t1))
    return y_explicit  # exact state


X = np.linspace(0, 1, 100)
t = np.linspace(0, 2, 200)

X_repeated = np.repeat(X, t.shape[0])
t_tiled = np.tile(t, X.shape[0])
XX = np.vstack((X_repeated, t_tiled)).T

state_predict = model.predict(XX).T
state_predict_M = state_predict.reshape((100, 200)).T

Xx, Tt = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 200))

fig = plt.figure()  # plot of predicted state
ax = plt.axes(projection="3d")
surf = ax.plot_surface(
    Xx, Tt, state_predict_M, cmap="hsv_r", linewidth=0, antialiased=False
)
ax.set_title("PINN state ")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.show()

state_exact = explicit_state(Xx, Tt)  # computation of exact state

fig = plt.figure()  # plot of exact state
ax = plt.axes(projection="3d")
surf2 = ax.plot_surface(
    Xx, Tt, state_exact, cmap="hsv_r", linewidth=0, antialiased=False
)

ax.set_title("Exact state ")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

fig.colorbar(surf2, shrink=0.6, aspect=10)

plt.show()

error_state = state_exact - state_predict_M  # error in state variable

fig, ax = plt.subplots()  # plot of error in state variable
norm = cm.colors.Normalize(vmax=abs(error_state).max(), vmin=-abs(error_state).max())
cont = ax.contourf(Xx, Tt, error_state, 40, norm=norm, cmap="hsv_r")

ax.set_title("Error between exact state and PINN state")
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

fig.colorbar(cont, shrink=1, aspect=10)

plt.show()
