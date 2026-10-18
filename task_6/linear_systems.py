import numpy as np

print("--- Task 6: Solution of Systems of Linear Equations ---")

# Consider a system of linear equations representing an electrical circuit or resource model:
# 3x + y + z = 2
# x + 4y + 2z = 12
# 2x + 2y + 5z = 10

# 1. Define coefficient matrix A and constants vector b
A = np.array([[3, 1, 1],
              [1, 4, 2],
              [2, 2, 5]], dtype=float)

b = np.array([2, 12, 10], dtype=float)

print("Coefficient Matrix A:\n", A)
print("Constants Vector b:", b)

# 2. Solve the linear system Ax = b using NumPy's linalg.solve
solution = np.linalg.solve(A, b)

print("\n--- Solution Vector x ---")
print(f"x = {solution[0]:.4f}")
print(f"y = {solution[1]:.4f}")
print(f"z = {solution[2]:.4f}")

# 3. Verification: Check if A @ x equals b
verification = np.dot(A, solution)
print(f"\nVerification (A @ x): {verification}")
print(f"Matches original b vector: {np.allclose(verification, b)}")