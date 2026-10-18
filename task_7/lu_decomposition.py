import numpy as np
from scipy.linalg import lu

print("--- Task 7: Gaussian Elimination and LU Decomposition ---")

# Define matrix A
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

print("Original Matrix A:\n", A)

# Perform LU Decomposition: A = P @ L @ U
# P: Permutation matrix, L: Lower triangular, U: Upper triangular
P, L, U = lu(A)

print("\nPermutation Matrix P:\n", P)
print("\nLower Triangular Matrix L:\n", L)
print("\nUpper Triangular Matrix U:\n", U)

# Verification: Reconstruct original matrix from P, L, and U
reconstructed_A = P @ L @ U
print("\nReconstructed Matrix (P @ L @ U):\n", reconstructed_A)
print(f"Matches original matrix: {np.allclose(A, reconstructed_A)}")