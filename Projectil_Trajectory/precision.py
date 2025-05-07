import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Simulated step sizes and errors
step_sizes = np.logspace(-4, -1, num=50)[::-1]
euler_errors = step_sizes * 50 + np.random.normal(0, 0.01, size=len(step_sizes))
rk4_errors = step_sizes**4 * 500 + np.random.normal(0, 1e-5, size=len(step_sizes))

# === FILTERING ===

# Euler: remove NaNs and non-positive values
valid_euler_indices = ~np.isnan(euler_errors) & (euler_errors > 0)
euler_errors_filtered = euler_errors[valid_euler_indices]
step_sizes_euler_filtered = step_sizes[valid_euler_indices]

# RK4: remove NaNs and non-positive values
valid_rk4_indices = ~np.isnan(rk4_errors) & (rk4_errors > 0)
rk4_errors_filtered = rk4_errors[valid_rk4_indices]
step_sizes_rk4_filtered = step_sizes[valid_rk4_indices]

# Remove outliers from RK4 using jump detection in log space
log_rk4 = np.log(rk4_errors_filtered)
diffs = np.abs(np.diff(log_rk4, prepend=log_rk4[0]))
smooth_indices = diffs < 2  # Filter out sharp log jumps
rk4_errors_final = rk4_errors_filtered[smooth_indices]
step_sizes_rk4_final = step_sizes_rk4_filtered[smooth_indices]

# === PLOT 1: LINEAR SCALE ===
plt.figure(figsize=(8, 6))
plt.plot(step_sizes_euler_filtered, euler_errors_filtered, 'o-', label='Euler (Filtered)')
plt.plot(step_sizes_rk4_final, rk4_errors_final, 's-', label='Runge-Kutta (RK4) Filtered')
plt.gca().invert_xaxis()
plt.xlabel('Step size (Δt)')
plt.ylabel('Global error (L2 norm)')
plt.title('Filtered Error vs Δt (Linear Scale)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === PLOT 2: LOG-LOG SCALE ===
plt.figure(figsize=(8, 6))
plt.loglog(step_sizes_euler_filtered, euler_errors_filtered, 'o-', label='Euler (Filtered)')
plt.loglog(step_sizes_rk4_final, rk4_errors_final, 's-', label='Runge-Kutta (RK4) Filtered')
plt.gca().invert_xaxis()
plt.xlabel('Step size (Δt)')
plt.ylabel('Global error (L2 norm)')
plt.title('Filtered Error vs Δt (Log-Log Scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()

# === SLOPE ESTIMATIONS ===

# Euler slope
log_dt_euler = np.log(step_sizes_euler_filtered)
log_euler = np.log(euler_errors_filtered)
slope_euler, _, _, _, _ = linregress(log_dt_euler, log_euler)

# RK4 full slope
log_dt_rk4 = np.log(step_sizes_rk4_final)
log_rk4 = np.log(rk4_errors_final)
slope_rk4_full, _, _, _, _ = linregress(log_dt_rk4, log_rk4)

# RK4 initial segment slope
N_initial = 5
slope_rk4_initial, _, _, _, _ = linregress(log_dt_rk4[:N_initial], log_rk4[:N_initial])

# === OUTPUT ===
print(f"Slope (Euler, full):      {slope_euler:.2f}")
print(f"Slope (RK4, full):        {slope_rk4_full:.2f}")
print(f"Slope (RK4, initial {N_initial}): {slope_rk4_initial:.2f}")


# Compute slope variation along RK4 error curve (sliding window)
window_size = 4
slope_variation_rk4 = []
slope_window_centers = []

for i in range(len(log_dt_rk4) - window_size + 1):
    x_window = log_dt_rk4[i:i+window_size]
    y_window = log_rk4[i:i+window_size]
    slope, _, _, _, _ = linregress(x_window, y_window)
    slope_variation_rk4.append(slope)
    slope_window_centers.append(np.mean(x_window))

# Similarly for Euler (optional smaller range)
slope_variation_euler = []
slope_window_centers_euler = []
for i in range(len(log_dt_euler) - window_size + 1):
    x_window = log_dt_euler[i:i+window_size]
    y_window = log_euler[i:i+window_size]
    slope, _, _, _, _ = linregress(x_window, y_window)
    slope_variation_euler.append(slope)
    slope_window_centers_euler.append(np.mean(x_window))

# Plot slope variation
plt.figure(figsize=(8, 6))
plt.plot(slope_window_centers_euler, slope_variation_euler, 'o-', label='Euler slope')
plt.plot(slope_window_centers, slope_variation_rk4, 's-', label='RK4 slope')
plt.xlabel('log(Δt) center')
plt.ylabel('Estimated slope')
plt.title(f'Slope Variation (Window Size = {window_size})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

