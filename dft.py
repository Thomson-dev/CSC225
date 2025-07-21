
import numpy as np
import matplotlib.pyplot as plt
import time

# Sampled signal
x = np.array([
    0.00000000e+00, 5.57590997e+00, 2.04087031e+00, -8.37717508e+00,
   -5.02028540e-01, 1.00000000e+01, -5.20431056e+00, -7.68722952e-01,
   -5.56758182e+00, 1.02781920e+01, 1.71450552e-15, -1.02781920e+01,
    5.56758182e+00, 7.68722952e-01, 5.20431056e+00, -1.00000000e+01,
    5.02028540e-01, 8.37717508e+00, -2.04087031e+00, -5.57590997e+00
])

# Time vector for plotting
t = np.linspace(0, 0.95, len(x))

# Plot the sampled signal
plt.figure(figsize=(10, 4))
plt.plot(t, x, 'o-', label='Sampled Signal')
plt.title('Sampled Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# ----------------------
# DFT Using Matrix Method with Timing
# ----------------------

N = len(x)
n = np.arange(N)
k = n.reshape((N, 1))

start_time = time.time()  # ⏱️ Start timing

W = np.exp(-2j * np.pi * k * n / N)  # DFT matrix
X = W @ x  # DFT computation
end_time = time.time()  # ⏱️ Stop timing

elapsed_time = end_time - start_time
print(f"Time to compute DFT with N={N}: {elapsed_time:.6f} seconds")

X_mag = np.abs(X)

# Frequency axis
fs = 20
freq = np.fft.fftfreq(N, d=1/fs)
positive_freqs = freq[:N//2]
positive_mag = X_mag[:N//2]

# Plot magnitude spectrum
plt.figure(figsize=(10, 4))
plt.stem(positive_freqs, positive_mag, basefmt=" ")
plt.title('Magnitude Spectrum (DFT using Matrix Method)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k]|')
plt.grid(True)
plt.show()

# Detect dominant frequencies
threshold = max(positive_mag) * 0.2
dominant_freqs = positive_freqs[positive_mag > threshold]

print("Estimated Frequencies (Hz):", dominant_freqs)
print("Estimated Number of Physical Systems:", len(dominant_freqs))
