import numpy as np

def my_cubic_spline_flat(x, y, X):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    X = np.array(X, dtype=float)

    n = len(x) - 1
    h = np.diff(x)  # h_i = x_{i+1} - x_i

    # Step 1: Construct the tridiagonal matrix A and right-hand side vector rhs
    A = np.zeros((n-1, n-1))
    rhs = np.zeros(n-1)

    for i in range(1, n):
        A[i-1, i-1] = 2 * (h[i-1] + h[i])
        if i - 2 >= 0:
            A[i-1, i-2] = h[i-1]
        if i < n - 1:
            A[i-1, i] = h[i]
        rhs[i-1] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    # Step 2: Solve the system for second derivatives M
    M = np.zeros(n + 1)
    if n > 1:
        M[1:n] = np.linalg.solve(A, rhs)

    # Step 3: Interpolate
    Y = np.zeros_like(X)

    for j, xj in enumerate(X):
        # Find the interval i such that x[i] <= xj <= x[i+1]
        i = np.searchsorted(x, xj) - 1
        if i < 0:
            i = 0
        elif i >= n:
            i = n - 1

        dx = xj - x[i]
        hi = h[i]

        a = y[i]
        b = (y[i+1] - y[i]) / hi - hi * (2 * M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * hi)

        Y[j] = a + b * dx + c * dx**2 + d * dx**3

    return Y


# ✅ Test Case
x = [0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8]
y = [10., 11.216, 11.728, 11.632, 11.024, 10., 8.656, 7.088, 5.392, 3.664,
     2., 0.496, -0.752, -1.648, -2.096]
X = [0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8]

Y = my_cubic_spline_flat(x, y, X)

# Print result
for xi, yi in zip(X, Y):
    print(f"S({xi:.1f}) = {yi:.6f}")
