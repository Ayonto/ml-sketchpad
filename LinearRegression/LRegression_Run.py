from LinearRegression import LinearRegression
import pandas as pd
import numpy as np 


df = pd.read_csv('LinearRegression/dataset.csv')

d = df.experience

# Convert Age & Experience to a NumPy array
x_train = df[['age', 'experience']].to_numpy()  # More efficient than .values.tolist()

# Convert Salary to a NumPy array
y_train = df['income'].to_numpy()



# model fitting
model = LinearRegression()
model.fit(x_train, y_train, iters = 100000)

print(model.predict([25,2]))
print(f"W: {model.w}")

#########################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example Data
X = x_train
Y = y_train

# Define W (weights) and b (bias) for the plane
W = model.w 
b = 100  # Example bias

# Generate a grid for the plane
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)  # Feature 1 range
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)  # Feature 2 range
X1, X2 = np.meshgrid(x1_range, x2_range)  # Create grid
Z = W[0] * X1 + W[1] * X2 + b  # Compute Y values using the equation

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data points
ax.scatter(X[:, 0], X[:, 1], Y, c='r', marker='o', label="Data Points")

# Plot the plane
ax.plot_surface(X1, X2, Z, alpha=0.5, color='cyan')

# Labels
ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Salary Output (Y)')
ax.set_title('3D Plane: Y = W * X + b')

plt.legend()
plt.show()
