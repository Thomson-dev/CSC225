import numpy as np

# Step 1: Doolittle LU Decomposition
def doolittle_lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i][i] = 1  # Diagonal of L is 1

        for j in range(i, n):  # Compute U
            U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
        
        for j in range(i+1, n):  # Compute L
            L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]
    
    return L, U

# Step 2: Forward substitution to solve Ly = b
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=float)
    
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y

# Step 3: Backward substitution to solve Ux = y
def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y, dtype=float)
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

# Define coefficient matrix A and right-hand side b
A = np.array([
    [1, -1, 3, 2],
    [-1, 5, -5, -2],
    [3, -5, 19, 3],
    [2, -2, 3, 21]
], dtype=float)

b = np.array([15, -35, 94, 1], dtype=float)

# Perform LU decomposition and solve
L, U = doolittle_lu_decomposition(A)
y = forward_substitution(L, b)
x = backward_substitution(U, y)

print("Solution vector x:")
for i, val in enumerate(x, start=1):
    print(f"x{i} = {val:.2f}")
