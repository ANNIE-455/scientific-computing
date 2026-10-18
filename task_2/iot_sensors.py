import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

print("--- IoT Sensor Data Numerical Computation ---")

# 1. Simulating IoT Sensor Data (e.g., 5 days of hourly temperature readings = 120 data points)
# Normal temperatures hovering around 25°C with some random noise
raw_temperatures = 25.0 + 2.5 * np.random.randn(120)

# Injecting a few artificial sensor glitches (outliers)
raw_temperatures[15] = 95.5   # Faulty spike
raw_temperatures[82] = -40.0  # Faulty drop

print(f"Total readings collected: {len(raw_temperatures)}")
print(f"First 10 raw readings: {np.round(raw_temperatures[:10], 2)}")

# 2. Vectorized Statistical Computation using NumPy
mean_temp = np.mean(raw_temperatures)
median_temp = np.median(raw_temperatures)
std_temp = np.std(raw_temperatures)
min_temp = np.min(raw_temperatures)
max_temp = np.max(raw_temperatures)

print("\n--- Raw Data Statistics ---")
print(f"Mean Temperature: {mean_temp:.2f} °C")
print(f"Median Temperature: {median_temp:.2f} °C")
print(f"Standard Deviation: {std_temp:.2f} °C")
print(f"Min Temperature: {min_temp:.2f} °C")
print(f"Max Temperature: {max_temp:.2f} °C")

# 3. Array Filtering & Cleaning (Removing Outliers)
# Filter realistic temperatures between 10°C and 40°C
valid_mask = (raw_temperatures >= 10.0) & (raw_temperatures <= 40.0)
cleaned_temperatures = raw_temperatures[valid_mask]

print(f"\nOutliers removed: {len(raw_temperatures) - len(cleaned_temperatures)}")
print(f"Cleaned Mean Temperature: {np.mean(cleaned_temperatures):.2f} °C")

# 4. Reshaping Data into a 2D Array (Days x Hours per day -> 5 days x 24 hours)
# We use only the first 120 valid or truncated elements if needed, or shape the cleaned subset
# Let's reshape the raw array into a 5 x 24 matrix to analyze daily cycles
daily_matrix = raw_temperatures.reshape((5, 24))
print("\n--- Daily Sensor Matrix Shape ---")
print(f"Matrix shape (Days, Hours): {daily_matrix.shape}")

# Compute daily average temperatures
daily_averages = np.mean(daily_matrix, axis=1)
for day_idx, avg in enumerate(daily_averages, start=1):
    print(f"Day {day_idx} Average Temperature: {avg:.2f} °C")