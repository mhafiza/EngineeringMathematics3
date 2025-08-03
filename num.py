# For any inquiries, please contact mhafiza@hotmail.com

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

#---------------------------------------#
#Newton-Raphson method for finding roots#
#---------------------------------------#

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):

    x = x0
    errors = []
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ZeroDivisionError(f"Derivative near zero at iteration {i}, x={x}")
        x_new = x - fx / dfx
        errors.append(abs(x_new - x))
        if abs(fx) < tol:
            x = x_new
            break
        x = x_new
    return x, i+1, errors

#----------------------------------#
#Bisection method for finding roots#
#----------------------------------#

def bisection(f, a, b, tol=1e-6, max_iter=100):

    if f(a) * f(b) >= 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    errors = []
    c = a  # Initialize c to prevent undefined if loop doesn’t run
    for i in range(max_iter):
        c = (a + b) / 2
        errors.append(abs(b - a))
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            break
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c, i+1, errors

#-------------------------------#
#Secant method for finding roots#
#-------------------------------#

def secant(f, x0, x1, tol=1e-6, max_iter=100):

    errors = []
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        denom = fx1 - fx0
        if abs(denom) < 1e-12:
            raise ZeroDivisionError(f"Zero division in secant method at iteration {i}")
        x_new = x1 - fx1 * (x1 - x0) / denom
        errors.append(abs(x_new - x1))
        if abs(fx1) < tol:
            x1 = x_new
            break
        x0, x1 = x1, x_new
    return x1, i+1, errors

# Example function and its derivative
def f(x):
    return x**3 - 2*x - 5

def df(x):
    return 3*x**2 - 2

# Initial parameters
a, b = 2, 3  # for bisection
x0 = 2.5     # for Newton-Raphson
x0_sec, x1_sec = 2, 3  # for secant

# Find roots using all three methods
nr_root, nr_iter, nr_errors = newton_raphson(f, df, x0)
bis_root, bis_iter, bis_errors = bisection(f, a, b)
sec_root, sec_iter, sec_errors = secant(f, x0_sec, x1_sec)

# Compare results
results = [
    ["Method", "Root", "Iterations", "Final Error"],
    ["Newton-Raphson", nr_root, nr_iter, nr_errors[-1] if nr_errors else 0],
    ["Bisection", bis_root, bis_iter, bis_errors[-1] if bis_errors else 0],
    ["Secant", sec_root, sec_iter, sec_errors[-1] if sec_errors else 0]
]

print(tabulate(results, headers="firstrow", tablefmt="grid", floatfmt=".10f"))

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(nr_errors)+1), nr_errors, 'o-', label='Newton-Raphson')
plt.semilogy(range(1, len(bis_errors)+1), bis_errors, 's-', label='Bisection')
plt.semilogy(range(1, len(sec_errors)+1), sec_errors, '^-', label='Secant')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Convergence Comparison of Root-Finding Methods')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
