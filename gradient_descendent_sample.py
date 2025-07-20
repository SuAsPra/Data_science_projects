import numpy as np
import matplotlib.pyplot as plt

# Function and its derivative
def f(x):
    return np.cos(2 * np.pi * x) + x**2

def df(x):
    return -2 * np.pi * np.sin(2 * np.pi * x) + 2 * x

# Gradient Descent Parameters
x_current = 0.5      # starting point
learning_rate = 0.01
iterations = 100

x_path = [x_current]  # to store the path
for i in range(iterations):
    grad = df(x_current)
    x_current -= learning_rate * grad
    x_path.append(x_current)

x_vals = np.linspace(-1.5, 1.5, 400)
y_vals = f(x_vals)
y_path = f(np.array(x_path))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = cos(2πx) + x²', color='blue')
plt.scatter(x_path, y_path, color='red', s=15, label='Gradient Descent Path')
plt.plot(x_path, y_path, color='red', linewidth=1)
plt.title('Gradient Descent on f(x) = cos(2πx) + x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
