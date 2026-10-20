import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Heat-balance model: f(T) = 0 means heat gain equals heat loss
def heat_balance(T):
    return T - 25.0 - 10.0 * np.exp(-T / 20.0)

# Bisection method using scipy
solution = root_scalar(
    heat_balance,
    bracket=[20, 40],
    method="bisect",
    xtol=1e-10
)

# Fixed-point iteration form: T = g(T) = 25 + 10*exp(-T/20)
def g(T):
    return 25.0 + 10.0 * np.exp(-T / 20.0)

T = 30.0
history = [T]
for _ in range(100):
    T_new = g(T)
    history.append(T_new)
    if abs(T_new - T) < 1e-10:
        T = T_new
        break
    T = T_new

print(f"Bisection equilibrium temperature: {solution.root:.6f} °C")
print(f"Fixed-point equilibrium temperature: {history[-1]:.6f} °C")
print(f"Fixed-point iterations taken: {len(history) - 1}")

# Visualization of equilibrium root
temps = np.linspace(20, 40, 300)
plt.figure(figsize=(8, 5))
plt.plot(temps, heat_balance(temps), label="f(T)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(solution.root, color="red", linestyle=":", label=f"Root ≈ {solution.root:.2f}°C")
plt.xlabel("Temperature (°C)")
plt.ylabel("Heat-Balance Function f(T)")
plt.title("Equilibrium Temperature Root Finding")
plt.legend()
plt.grid(True)
plt.show()