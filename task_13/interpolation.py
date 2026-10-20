import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

# Hourly temperature measurements
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

# Estimate every 10 minutes = 1/6 hour
ten_minute_times = np.arange(8, 14 + 1/6, 1/6)
linear_estimates = linear(ten_minute_times)
cubic_estimates = cubic(ten_minute_times)

target_time = 10.5  # 10:30
print(f"Linear estimate at 10:30: {float(linear(target_time)):.2f} °C")
print(f"Cubic estimate at 10:30: {float(cubic(target_time)):.2f} °C")

plt.figure(figsize=(8, 5))
plt.scatter(hours, temperature, label="Hourly measurements", color="red")
plt.plot(ten_minute_times, linear_estimates, label="Linear Interpolation", linestyle="--")
plt.plot(ten_minute_times, cubic_estimates, label="Cubic Spline", linestyle="-")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Interpolation")
plt.legend()
plt.grid(True)
plt.show()