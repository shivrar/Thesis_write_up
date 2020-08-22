#!/usr/bin/env python
"""
This is a simple script file to read in pozyx tag position data and apply and ekf on it
since we're operating on a surface this will be a simple 2 state system.
"""

import pandas as pd
import numpy as np
import math
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

def fx(x, dt):
    return x

# Since we get X and Y the observation model is just the identity matrix
def H_of(x):
    return np.eye(2, 2)

def H_x(x):
    return x

def F_of(x):
    return np.eye(2, 2)


# We can update the measurement noise
def R_unc(x_e, y_e, xy_e):
    return np.array([[x_e, xy_e], [xy_e, y_e]])


if __name__ == "__main__":
    data = pd.read_csv("for_thesis/path_traingle_person_random.csv", skiprows=1)

    # time = data['time(s)'].to_numpy()
    x_measured = data['x'].to_numpy()
    y_measured = data['y'].to_numpy()
    z_measured = data['z'].to_numpy()

    x_err = data['x_err'].to_numpy()
    y_err = data['y_err'].to_numpy()
    xy_err = data['xy_err'].to_numpy()

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=2)
    ekf.x = np.array([x_measured[0], y_measured[0]])
    ekf.R = np.diag([0.1, 0.1])
    ekf.predict()
    estimates = []

    for i in range(1, x_measured.shape[0], 1):
        if math.sqrt((x_measured[i] - x_measured[i-1])**2 + (y_measured[i] - y_measured[i])**2) < 30:
            measurements = np.array([x_measured[i], y_measured[i]])
            ekf.update(z=measurements, HJacobian=H_of, Hx=H_x, R=R_unc(x_err[i], y_err[i], xy_err[i]))
            ekf.predict()
            estimates.append(ekf.x)

    ests = np.array(estimates)
    triangle_x = [1610, 2111, 1910, 1610, 1610]
    triangle_y = [2080, 2080, 2380, 2580, 2080]

    c_x = [1610, 1710, 1760, 1770, 1840, 1934, 1992]
    c_y = [2080, 2200, 2300, 2500, 2580, 2625, 2595]
    # print(ests.shape)
    # plt.plot(ests[:, 0], ests[:, 1], x_measured, y_measured, c_x, c_y, 'r')
    plt.plot(ests[:, 0], ests[:, 1], x_measured, y_measured, triangle_x, triangle_y, 'r')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('EKF vs Measured vs Path')
    plt.legend(['Estimated', 'Measured', 'Path outline'])
    plt.grid()
    plt.show()


    # print(x_measured[0], y_measured[0])