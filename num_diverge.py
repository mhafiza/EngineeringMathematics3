# For any inquiries, please contact mhafiza@hotmail.com

import numpy as np
import matplotlib.pyplot as plt

# Define function and its derivative
def f(x):
    return x**3 - 2*x + 2

def df(x):
    return 3*x**2 - 2

# Newton-Raphson method
def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):
    errors = []
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            print(f"Iteration {i}: derivative too small; possible divergence.")
            break
        x_new = x - fx / dfx
        errors.append(abs(x_new - x))
        if abs(fx) < tol:
            x = x_new
            break
        x = x_new
    return x, i+1, errors

# Secant method
def secant(f, x0, x1, tol=1e-6, max_iter=50):
    errors = []
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        denom = fx1 - fx0
        if abs(denom) < 1e-12:
            print(f"Iteration {i}: denominator too small.")
            break
        x_new = x1 - fx1 * (x1 - x0) / denom
        errors.append(abs(x_new - x1))
        if abs(fx1) < tol:
            x1 = x_new
            break
        x0, x1 = x1, x_new
    return x1, i+1, errors

# Bisection method
def bisection(f, a, b, tol=1e-6, max_iter=50):
    if f(a)*f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    errors = []
    for i in range(max_iter):
        c = (a + b)/2
        errors.append(abs(b - a))
        if abs(f(c)) < tol or (b - a)/2 < tol:
            break
        if f(c)*f(a) < 0:
            b = c
        else:
            a = c
    return c, i+1, errors

# Initial parameters
x0_nr = 0.0      # bad initial guess for Newton-Raphson
x0_sec, x1_sec = -2, -1  # Secant guesses spanning the root
a, b = -2, -1    # Bisection interval containing sign change

# Run methods
print("Running Newton-Raphson method...")
nr_root, nr_iter, nr_errors = newton_raphson(f, df, x0_nr)

print("Running Secant method...")
sec_root, sec_iter, sec_errors = secant(f, x0_sec, x1_sec)

print("Running Bisection method...")
bis_root, bis_iter, bis_errors = bisection(f, a, b)

# Show results
print("\nResults:")
print(f"Newton-Raphson:  root ≈ {nr_root:.6f}, iterations = {nr_iter}")
print(f"Secant:          root ≈ {sec_root:.6f}, iterations = {sec_iter}")
print(f"Bisection:       root ≈ {bis_root:.6f}, iterations = {bis_iter}")

# Plot convergence
plt.figure(figsize=(10,6))
if nr_errors: plt.semilogy(range(1,len(nr_errors)+1), nr_errors, 's-', label='Newton-Raphson')
if sec_errors: plt.semilogy(range(1,len(sec_errors)+1), sec_errors, '^-', label='Secant')
if bis_errors: plt.semilogy(range(1,len(bis_errors)+1), bis_errors, 'o-', label='Bisection')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Convergence comparison on $f(x)=x^3 - 2x + 2$')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
