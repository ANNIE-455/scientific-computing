import numpy as np

print("--- Task 3: Mathematical Functions and Vectorization ---")

# 1. Generate a time vector (e.g., 1000 time steps over 2 seconds)
time = np.linspace(0, 2, 1000)

# 2. Create a clean signal: Sine wave of 5 Hz + Cosine wave of 10 Hz
clean_signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.cos(2 * np.pi * 10 * time)

# 3. Add random Gaussian noise (simulating sensor noise/interference)
np.random.seed(42)
noise = 0.2 * np.random.randn(len(time))
noisy_signal = clean_signal + noise

# 4. Vectorized Mathematical Operations (Filtering/Transforming the signal)
# Apply an exponential damping function to simulate signal attenuation over time
damped_signal = noisy_signal * np.exp(-0.5 * time)

# Apply a non-linear activation/transformation (e.g., hyperbolic tangent) common in neural networks
transformed_signal = np.tanh(damped_signal)

# 5. Compute summary metrics using vectorized reductions
max_amplitude = np.max(transformed_signal)
min_amplitude = np.min(transformed_signal)
rms_value = np.sqrt(np.mean(transformed_signal ** 2)) # Root Mean Square

print(f"Total signal data points: {len(time)}")
print(f"Max Amplitude after vector processing: {max_amplitude:.4f}")
print(f"Min Amplitude after vector processing: {min_amplitude:.4f}")
print(f"RMS (Root Mean Square) Value: {rms_value:.4f}")