# DFT Analysis of Sampled Signal
# This script performs a Discrete Fourier Transform (DFT) analysis on a sampled signal,

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 1: Define sampled signal and sampling info
x = np.array([
    0.00000000e+00, 5.57590997e+00, 2.04087031e+00, -8.37717508e+00, -5.02028540e-01,
    1.00000000e+01, -5.20431056e+00, -7.68722952e-01, -5.56758182e+00, 1.02781920e+01,
    1.71450552e-15, -1.02781920e+01, 5.56758182e+00, 7.68722952e-01, 5.20431056e+00,
    -1.00000000e+01, 5.02028540e-01, 8.37717508e+00, -2.04087031e+00, -5.57590997e+00
])
t = np.arange(0, 1.0, 0.05)  # Time vector (20 samples from 0 to 0.95s)
Fs = 20  # Sampling frequency in Hz
N = len(x)

# Step 2: Plot the time-domain signal
plt.figure(figsize=(10, 4))
plt.plot(t, x, marker='o')
plt.title("Sampled Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Compute DFT and frequencies
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/Fs)

# Step 4: Use only the positive frequencies
half_N = N // 2
X_mag = np.abs(X[:half_N])
freqs_pos = freqs[:half_N]

# Step 5: Plot the magnitude spectrum
plt.figure(figsize=(10, 4))
plt.stem(freqs_pos, X_mag)
plt.title("Magnitude Spectrum (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Detect and print peak frequencies (physical systems)
peaks, _ = find_peaks(X_mag, height=1)  # height threshold removes noise
peak_freqs = freqs_pos[peaks]
peak_mags = X_mag[peaks]

print("📡 Detected Physical Systems:")
for f, mag in zip(peak_freqs, peak_mags):
    print(f"→ Frequency: {f:.1f} Hz, Magnitude: {mag:.2f}")
