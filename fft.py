
import numpy as np
import matplotlib.pyplot as plt
import time

# Given sampled signal
x = np.array([
    0.00000000e+00, 5.57590997e+00, 2.04087031e+00, -8.37717508e+00,
   -5.02028540e-01, 1.00000000e+01, -5.20431056e+00, -7.68722952e-01,
   -5.56758182e+00, 1.02781920e+01, 1.71450552e-15, -1.02781920e+01,
    5.56758182e+00, 7.68722952e-01, 5.20431056e+00, -1.00000000e+01,
    5.02028540e-01, 8.37717508e+00, -2.04087031e+00, -5.57590997e+00
])

fs = 20  # Sampling frequency in Hz
N = len(x)
t = np.linspace(0, (N-1)/fs, N)

# Plot the sampled signal
plt.figure(figsize=(10,4))
plt.plot(t, x, 'o-', label='Sampled Signal')
plt.title('Sampled Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Measure execution time for FFT
start_time = time.time()
X_fft = np.fft.fft(x)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"\nFFT Computation Time for N={N}: {elapsed_time:.8f} seconds")

# Frequency axis
freq = np.fft.fftfreq(N, d=1/fs)

# Take only positive frequencies (first half)
half_N = N // 2
freq_pos = freq[:half_N]
X_mag = np.abs(X_fft[:half_N])

# Plot magnitude spectrum
plt.figure(figsize=(10,4))
plt.stem(freq_pos, X_mag, basefmt=" ")
plt.title('Magnitude Spectrum (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k]|')
plt.grid(True)
plt.show()

# Detect dominant frequencies (above 20% threshold)
threshold = 0.2 * np.max(X_mag)
dominant_freqs = freq_pos[X_mag > threshold]

print("\nEstimated Frequencies (Hz):", dominant_freqs)
print("Estimated Number of Physical Systems:", len(dominant_freqs))

