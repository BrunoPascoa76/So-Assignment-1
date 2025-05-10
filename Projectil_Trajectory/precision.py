import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import euler
import runge_kutta
import numerical

# Step sizes (logarithmic scale)
step_sizes = np.logspace(-4, -1, num=1000)[::-1]

euler_errors = []
rk4_errors = []

# Base simulation parameters (constant across all runs)
params_base = dict(x0=10, z0=5, vx0=70, vz0=30, u_val=3, m_val=1, g_val=9.8, t_final=10000)

# Run simulations for each Δt
for delta in step_sizes:
    params = params_base.copy()
    params["delta"] = delta

    ex, ez, *_ = euler.main(**params, if_plot=False)
    rx, rz, *_ = runge_kutta.main(**params, if_plot=False)
    nx, nz, *_ = numerical.main(**params, if_plot=False)

    # Align lengths and compute error
    ex, ez, rx, rz, nx, nz = map(np.array, [ex, ez, rx, rz, nx, nz])
    N = min(len(ex), len(rx), len(nx))

    euler_errors.append(np.linalg.norm(ex[:N] - nx[:N]) + np.linalg.norm(ez[:N] - nz[:N]))
    rk4_errors.append(np.linalg.norm(rx[:N] - nx[:N]) + np.linalg.norm(rz[:N] - nz[:N]))

# Convert to arrays
step_sizes = np.array(step_sizes)
euler_errors = np.array(euler_errors)
rk4_errors = np.array(rk4_errors)

# === FILTERING ===

# Euler: remove NaNs and non-positive values
valid_euler = (euler_errors > 0) & np.isfinite(euler_errors)
step_sizes_euler_filtered = step_sizes[valid_euler]
euler_errors_filtered = euler_errors[valid_euler]

# RK4: remove NaNs and non-positive values
valid_rk4 = (rk4_errors > 0) & np.isfinite(rk4_errors)
step_sizes_rk4_filtered = step_sizes[valid_rk4]
rk4_errors_filtered = rk4_errors[valid_rk4]

# Further filter RK4 for sudden log jumps
log_rk4 = np.log(rk4_errors_filtered)
diffs = np.abs(np.diff(log_rk4, prepend=log_rk4[0]))
smooth_rk4 = diffs < 2
rk4_errors_final = rk4_errors_filtered[smooth_rk4]
step_sizes_rk4_final = step_sizes_rk4_filtered[smooth_rk4]

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
log_dt_euler = np.log(step_sizes_euler_filtered)
log_err_euler = np.log(euler_errors_filtered)
slope_euler, _, _, _, _ = linregress(log_dt_euler, log_err_euler)

log_dt_rk4 = np.log(step_sizes_rk4_final)
log_err_rk4 = np.log(rk4_errors_final)
slope_rk4_full, _, _, _, _ = linregress(log_dt_rk4, log_err_rk4)

N_final = 100
slope_euler_final, _, _, _, _ = linregress(
    log_dt_euler[len(log_dt_euler)-N_final:len(log_dt_euler)], log_err_euler[len(log_dt_euler)-N_final:len(log_dt_euler)]
)
slope_rk4_final, _, _, _, _ = linregress(
    log_dt_rk4[len(log_dt_rk4)-N_final:len(log_dt_rk4)], log_err_rk4[len(log_dt_rk4)-N_final:len(log_dt_rk4)]
)
print(f"Slope (Euler, full):             {slope_euler:.2f}")
print(f"Slope (RK4, full):               {slope_rk4_full:.2f}")
print(f"Slope (Euler, final {N_final}): {slope_euler_final:.2f}")
print(f"Slope (RK4, final {N_final}): {slope_rk4_final:.2f}")

# === SLOPE PLOT ===
slope_euler_list = []
slope_rk4_list = []

for k in range(2, len(log_dt_euler)):
    slope_euler_k, _, _, _, _ = linregress(
        log_dt_euler[len(log_dt_euler)-k:len(log_dt_euler)], log_err_euler[len(log_dt_euler)-k:len(log_dt_euler)]
    )
    slope_euler_list.append(slope_euler_k)

for k in range(2, len(log_dt_rk4)):
    slope_rk4_k, _, _, _, _ = linregress(
        log_dt_rk4[len(log_dt_rk4)-k:len(log_dt_rk4)], log_err_rk4[len(log_dt_rk4)-k:len(log_dt_rk4)]
    )
    slope_rk4_list.append(slope_rk4_k)

plt.figure(figsize=(8, 6))
plt.plot(log_dt_euler[-len(slope_euler_list):], slope_euler_list, label='Euler slope')
plt.plot(log_dt_rk4[-len(slope_rk4_list):], slope_rk4_list, label='RK4 slope')
plt.xlabel("log(Δt)")
plt.ylabel("slope")
plt.title("Variation of slope (from end to beginning)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# === SLIDING WINDOW SLOPE VARIATION ===
window = 100

def compute_sliding_slope(log_x, log_y, w):
    slopes = []
    centers = []
    for i in range(len(log_x) - w + 1):
        x_window = log_x[i:i + w]
        y_window = log_y[i:i + w]
        slope, _, _, _, _ = linregress(x_window, y_window)
        slopes.append(slope)
        centers.append(np.mean(x_window))
    return np.array(slopes), np.array(centers)

euler_slopes, euler_centers = compute_sliding_slope(log_dt_euler, log_err_euler, window)
rk4_slopes, rk4_centers = compute_sliding_slope(log_dt_rk4, log_err_rk4, window)

# Plot: slope variation vs log(Δt)
plt.figure(figsize=(8, 6))
plt.plot(euler_centers, euler_slopes, 'o-', label='Euler local slope')
plt.plot(rk4_centers, rk4_slopes, 's-', label='RK4 local slope')
plt.xlabel("log(Δt)")
plt.ylabel("Estimated slope")
plt.title(f"Slope Variation of log(Error) vs log(Δt) (Window = {window})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

