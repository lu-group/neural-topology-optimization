# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:12:26 2024

@author: kouch
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rc('font',family='Times New Roman',size='20',weight='bold') 

Y = np.array([1,    0.5,   0.2,  0.1,  0.05,    0.01, 0]) #gray value
X = np.array([7.96, 18.82, 52.4, 85.8, 153.3,  208.2, 255]) #concentration value

# Filter for X > 5 and Y >= 0
mask = (X > 0) & (Y >= 0)
X_positive = X[mask]
Y_positive = Y[mask]

# Decreasing power function
def decreasing_power_func(x, a, b,c):
    return a * (x ** -b)-c

# Fit the model to the positive data
params, _ = curve_fit(decreasing_power_func, X_positive, Y_positive)

# Generate fitted values
fitted_Y = decreasing_power_func(X_positive, *params)

# Plotting
plt.scatter(X, Y, label='Experiment data', color='blue', marker='o')
plt.plot(X_positive, fitted_Y, label='Fitted Curve', color='red')
plt.yticks([0.0,0.5,1.0])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$C$')

plt.legend()
plt.grid(False)
plt.show()

# Print parameters
a, b,c = params
print(f"Fitted parameters: a = {a}, b = {b}, c = {c}")
