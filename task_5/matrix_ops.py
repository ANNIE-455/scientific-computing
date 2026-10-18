import numpy as np

print("--- Task 5: Matrix Algebra and Operations ---")

# 1. Define Matrices (e.g., transformation or network weights)
A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])

B = np.array([[4, 2],
              [1, 5],
              [3, 6]])

print("Matrix A (3x3):\n", A)
print("\nMatrix B (3x2):\n", B)

# 2. Matrix Multiplication (Dot Product): A * B (3x3 dot 3x2 = 3x2)
product_AB = np.dot(A, B) # or A @ B
print("\nDot Product of A and B (A @ B):\n", product_AB)

# 3. Transpose of Matrix A
transpose_A = A.T
print("\nTranspose of Matrix A:\n", transpose_A)

# 4. Determinant of Matrix A
det_A = np.linalg.det(A)
print(f"\nDeterminant of Matrix A: {det_A:.2f}")

# 5. Inverse of Matrix A (if determinant is non-zero)
if det_A != 0:
    inv_A = np.linalg.inv(A)
    print("\nInverse of Matrix A:\n", inv_A)
    
    # Verification: A dot inv_A should yield the Identity Matrix
    identity_check = np.dot(A, inv_A)
    print("\nVerification (A @ inv_A -> Identity Matrix):\n", np.round(identity_check))
else:
    print("\nMatrix A is singular and cannot be inverted.")