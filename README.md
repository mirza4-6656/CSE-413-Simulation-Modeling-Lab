# CSE-413-Simulation-Modeling-Lab
-----------------------------------------------------------------------------------------DAY-1--------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
import seaborn as sns

1. Scalar & Array Operations (Redo Section with Your Own Values):
Run the given code with scalar = 3.7 and a new array of at least 6 elements.

Show outputs of all built-in mathematical functions (sin, cos, tan, etc.).

scaler = 3.7
array = np.array([5, 6, 9, 8, 10, 7, 4])

# Trigonometric functions
sin_values = np.sin(array)
cos_values = np.cos(array)
tan_values = np.tan(array)
asin_values = np.arcsin(np.clip(array / np.max(array), -1, 1))  # Normalized and Clipping for valid range [-1, 1]
acos_values = np.arccos(np.clip(array / np.max(array), -1, 1))
atan_values = np.arctan(array)

# Display results
print("Resul of Sin:", sin_values)
print("Resul of Cos:", cos_values)
print("Resul of Tan:", tan_values)
print("Resul of asin:", asin_values)
print("Resul of acos:", acos_values)
print("Resul of atan:", atan_values)

Apply the Same Operations on a 3×2 Matrix
Create a 3×2 matrix with custom values.

Perform and display results of:
Trigonometric functions
Exponential & logarithmic functions
Absolute value & square root
Remainder when divided by 2
Rounding (round, floor, ceil)

# Creating a 3x2 matrix
matrix = np.array([[10, 2,], [42, 34], [59, 60]])

# Trigonometric functions
sin_values = np.sin(matrix)
cos_values = np.cos(matrix)
tan_values = np.tan(matrix)
asin_values = np.arcsin(np.clip(matrix / np.max(matrix), -1, 1))  # Clipping for valid range [-1, 1]
acos_values = np.arccos(np.clip(matrix / np.max(matrix), -1, 1))
atan_values = np.arctan(matrix)

# Exponential and logarithm
exp_values = np.exp(matrix)
log_values = np.log(matrix)  # Natural logarithm (log base e)

# Absolute value and square root
abs_values = np.abs(matrix)
sqrt_values = np.sqrt(matrix)

# Remainder when divided by a scalar (e.g., 2)
rem_values = np.remainder(matrix, 2)

# Rounding functions
round_values = np.round(matrix)
floor_values = np.floor(matrix)
ceil_values = np.ceil(matrix)

# Display results
print("Matrix:\n", matrix)
print("\nsin:\n", sin_values)
print("\ncos:\n", cos_values)
print("\ntan:\n", tan_values)
print("\nasine:\n", asin_values)
print("\nacosine:\n", acos_values)
print("\natangent:\n", atan_values)
print("\nexp:\n", exp_values)
print("\nlog (natural):\n", log_values)
print("\nabs:\n", abs_values)
print("\nsqrt:\n", sqrt_values)
print("\nrem (remainder when divided by 2):\n", rem_values)
print("\nround:\n", round_values)
print("\nfloor:\n", floor_values)
print("\nceil:\n", ceil_values)

Perform Same Operations on a 2×3 Matrix
Your 2×3 matrix can have float values (e.g., 2.1, 4.7…).

Implement and display:

Max & Min (with indices)

Vector length

Sorted version (both row-wise descending and entire matrix descending)

Sum, Product

Median, Mean, Standard Deviation

# Sample vector 2x3 matrix
vec = np.array([[3.3, 1.2, 4.6], [1.9, 5.2, 9.7]])

# 1. Maximum value and index of max element
max_value = np.max(vec)
max_index = np.argmax(vec)  # Index of max element

# 2. Minimum value and index of min element
min_value = np.min(vec)
min_index = np.argmin(vec)  # Index of min element

# 3. Length of the vector
vec_length = len(vec)  # Equivalent to MATLAB's length()

# 4. Sorting in ascending order
sorted_vec = np.sort(vec)

# 5. Sum of elements
sum_values = np.sum(vec)

# 6. Product of elements
prod_values = np.prod(vec)

# 7. Median value
median_value = np.median(vec)

# 8. Mean value
mean_value = np.mean(vec)

# 9. Standard deviation
std_dev = np.std(vec)

# Display results
print("Max value:", max_value, "at index", max_index)
print("Min value:", min_value, "at index", min_index)
print("Length of vector:", vec_length)
print("Sorted vector:", sorted_vec)
print("Sum of elements:", sum_values)
print("Product of elements:", prod_values)
print("Median value:", median_value)
print("Mean value:", mean_value)
print("Standard deviation:", std_dev)

-----------------------------------------------------------------------------------------DAY-2--------------------------------------------------------------------------------------------------------------------------------

d1 = 16  # last digit
d2 = 10  # second last digit

A = np.array([[d1 + 2, d2 + 1],
              [2*d1, d2 + 2]])

print("Matrix A:\n", A)

# Shape
shape_A = A.shape

# Determinant
det_A = np.linalg.det(A)

# Rank
rank_A = np.linalg.matrix_rank(A)

# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# Inverse (only if determinant != 0)
if det_A != 0:
    inv_A = np.linalg.inv(A)
else:
    inv_A = "Matrix is singular, cannot compute inverse"

# Display results
print("Shape of A:", shape_A)
print("Determinant of A:", det_A)
print("Rank of A:", rank_A)
print("Eigenvalues of A:", eigenvalues)
print("Eigenvectors of A:\n", eigenvectors)

B = A.copy()
B[0, 0] = B[0, 0] + 1

print("Matrix B:\n", B)
print("Inverse of A:\n", inv_A)

# Shape
shape_B = B.shape

# Determinant
det_B = np.linalg.det(B)

# Rank
rank_B = np.linalg.matrix_rank(B)

# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(B)

# Inverse (only if determinant != 0)
if det_B != 0:
    inv_B = np.linalg.inv(B)
else:
    inv_B = "Matrix is singular, cannot compute inverse"

# Display results
print("Shape of B:", shape_B)
print("Determinant of B:", det_B)
print("Rank of B:", rank_B)
print("Eigenvalues of B:", eigenvalues)
print("Eigenvectors of B:\n", eigenvectors)
print("Inverse of B:\n", inv_B)

1. How did the determinant change and why?
The determinant changed from −136 to −124. This happened because increasing one element alters the area-scaling factor of the matrix. Since determinants depend linearly on matrix entries, even a +1 change shifts the determinant.

2. Did the rank change?
No.Both matrices have rank 2. The rows and columns remain linearly independent, so the rank is unchanged.

3. How did the eigenvalues respond to the value change?
The eigenvalues changed slightly: The larger eigenvalue increased and the smaller which is negative eigenvalue moved closer to zero.This is expected because eigenvalues are sensitive to matrix entry changes, but small value changes usually cause small eigenvalue shifts, not drastic jumps.

4. Is B easier or harder to invert than A? Why?
B is slightly easier to invert than A because the absolute determinant of B (124) is smaller than A (136), but both are far from zero. Since neither matrix is close to being singular, inversion is stable in both cases.

-----------------------------------------------------------------------------------------DAY-3--------------------------------------------------------------------------------------------------------------------------------

# Create two random 3x3 matrices with integer values between 0 and 10
A = np.random.randint(1, 15, size=(5, 5))  # Random integers between 0 and 9
B = np.random.randint(2, 18, size=(5, 5))  # Random integers between 0 and 9

print("Matrix A:\n", A)
print("Matrix B:\n", B)

# Matrix operations
matrix_sum = A + B  # Matrix addition
matrix_diff = A - B  # Matrix subtraction
matrix_prod = np.dot(A, B)  # Matrix multiplication (dot product)

# 2. Determinant of a matrix
det_A = np.linalg.det(A)  # Returns the determinant of the matrix
det_B = np.linalg.det(B)
# 3. Inverse of a matrix
if np.linalg.det(A) != 0:  # Check if matrix is invertible
    inv_A = np.linalg.inv(A)  # Compute the inverse of A
else:
    inv_A = "Matrix is singular, cannot compute inverse"  # No inverse if determinant is 0
if np.linalg.det(B) != 0:  # Check if matrix is invertible
    inv_B = np.linalg.inv(B)  # Compute the inverse of A
else:
    inv_B = "Matrix is singular, cannot compute inverse"  # No inverse if determinant is 0

# 4. Rank of a matrix
rank_A = np.linalg.matrix_rank(A)  # Returns the rank of the matrix
rank_B = np.linalg.matrix_rank(B)

print("Addition of Matrix:\n", matrix_sum)
print("Subtraction of Matrix:\n", matrix_diff)
print("Multiplication of Matrix:\n", matrix_prod)
print("Determinant of A:", det_A)
print("Inverse of A:\n", inv_A)
print("Rank of A:", rank_A)
print("Determinant of B:", det_B)
print("Inverse of B:\n", inv_B)
print("Rank of B:", rank_B)

# Create two vectors of 10 random values each
x = np.random.rand(15) #by default [0,1) which includes 0 but excludes 1
y = np.random.rand(15)

# Create a scatter plot
plt.scatter(x, y, color='c', label='Random Points')
plt.title('Comparison of Two Random Vectors')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.grid(True)
plt.show()

# Generate a 4x4 matrix with random values
matrix = np.random.rand(4, 4) #by default [0,1) which includes 0 but excludes 1
print(matrix)
# Plot the heatmap
sns.heatmap(matrix, annot=True, cmap='spring', linewidths=0.5, linecolor ='black')
plt.title('Heatmap of 4x4 Matrix')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# Create two random 3x3 matrices with integer values between 0 and 10
A = np.random.randint(0, 10, size=(4, 4))  # Random integers between 0 and 9
B = np.random.randint(0, 10, size=(4, 4))  # Random integers between 0 and 9

print("Matrix A:\n", A)
print("Matrix B:\n", B)

# Matrix operations
matrix_sum = A + B  # Matrix addition
matrix_diff = A - B  # Matrix subtraction
matrix_prod = np.dot(A, B)  # Matrix multiplication (dot product)

# Bar plot to visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5)) #figure object, subplot array

# Sum
axes[0].bar(range(1, 17), matrix_sum.flatten(), color='b')
axes[0].set_title('Matrix Addition')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Value')

# Difference
axes[1].bar(range(1, 17), matrix_diff.flatten(), color='g')
axes[1].set_title('Matrix Subtraction')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Value')

# Product
axes[2].bar(range(1, 17), matrix_prod.flatten(), color='r')
axes[2].set_title('Matrix Multiplication')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Value')

plt.tight_layout() #plt.tight_layout() is a Matplotlib function that automatically adjusts the layout of subplots and figures to prevent overlapping and ensure a visually appealing presentation. It modifies the spacing between subplots, axes, labels, and titles to improve readability.
plt.show()

-----------------------------------------------------------------------------------------DAY-4--------------------------------------------------------------------------------------------------------------------------------
