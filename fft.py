# FFT Analysis of Sampled Signal

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

# Step 1: Define the sampled signal
x = np.array([
    0.00000000e+00, 5.57590997e+00, 2.04087031e+00, -8.37717508e+00, -5.02028540e-01,
    1.00000000e+01, -5.20431056e+00, -7.68722952e-01, -5.56758182e+00, 1.02781920e+01,
    1.71450552e-15, -1.02781920e+01, 5.56758182e+00, 7.68722952e-01, 5.20431056e+00,
    -1.00000000e+01, 5.02028540e-01, 8.37717508e+00, -2.04087031e+00, -5.57590997e+00
])
t = np.arange(0, 1.0, 0.05)  # Time vector
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

# Step 3: Compute FFT and timing
start_fft = time.time()
X_fft = np.fft.fft(x)
end_fft = time.time()

# Step 4: Compute magnitude and frequency axis
freqs = np.fft.fftfreq(N, d=1/Fs)
half_N = N // 2
X_mag = np.abs(X_fft[:half_N])
freqs_pos = freqs[:half_N]

# Step 5: Plot magnitude spectrum
plt.figure(figsize=(10, 4))
plt.stem(freqs_pos, X_mag)
plt.title("FFT Magnitude Spectrum (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Detect peak frequencies (physical systems)
peaks, _ = find_peaks(X_mag, height=1)  # Set height threshold to avoid noise
peak_freqs = freqs_pos[peaks]
peak_mags = X_mag[peaks]

print("📡 Detected Physical Systems using FFT:")
for f, mag in zip(peak_freqs, peak_mags):
    print(f"→ Frequency: {f:.1f} Hz, Magnitude: {mag:.2f}")

# Step 7: Report FFT execution time
print(f"\n⏱ FFT Execution Time: {end_fft - start_fft:.6f} seconds (Expected: O(N log N))")
