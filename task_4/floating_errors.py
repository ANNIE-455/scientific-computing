import numpy as np
import math

print("--- Task 4: Floating-Point Arithmetic and Numerical Errors ---")

# 1. The Classic Floating-Point Precision Issue
val = 0.1 + 0.2
print(f"0.1 + 0.2 = {val}")
print(f"Is 0.1 + 0.2 exactly equal to 0.3? {val == 0.3}")

# 2. Accumulation of Round-off Errors in Loops
# Adding 0.1 ten times vs multiplying by 1.0
sum_loop = 0.0
for _ in range(10):
    sum_loop += 0.1

print(f"Summing 0.1 ten times: {sum_loop}")
print(f"Direct multiplication (1.0 * 0.1 * 10): {1.0 * 0.1 * 10}")
print(f"Difference due to floating-point error: {abs(sum_loop - 1.0)}")

# 3. Safe Comparison using Tolerances (Epsilon / isclose)
# Standard naive check (fails due to rounding precision)
if sum_loop == 1.0:
    print("Naive check: Equal")
else:
    print("Naive check: NOT Equal (Precision Error detected!)")

# Correct robust check using math.isclose
is_safe_equal = math.isclose(sum_loop, 1.0, rel_tol=1e-9, abs_tol=1e-9)
print(f"Robust check using math.isclose(): {is_safe_equal}")

# 4. Vectorized error handling with NumPy
array_sums = np.full(1000, 0.1)
total_sum = np.sum(array_sums)
expected = 100.0

# Using np.isclose for array evaluations
close_check = np.isclose(total_sum, expected)
print(f"Array sum total: {total_sum}")
print(f"NumPy vectorized tolerance check passed: {close_check}")