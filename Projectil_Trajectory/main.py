import numpy as np
import matplotlib.pyplot as plt

import euler
import runge_kutta
import numerical


def run_simulation(x0=10, z0=5, vx0=70, vz0=30, u_val=3, m_val=1, g_val=9.8, delta=1e-3, t_final=10000):
    # Parameters
    params = dict(x0=x0, z0=z0, vx0=vx0, vz0=vz0, u_val=u_val, m_val=m_val, g_val=g_val, delta=delta, t_final=t_final)

    # Run methods
    ex, ez, evx, evz, et = euler.main(**params, if_plot=False)
    rx, rz, rvx, rvz, rt = runge_kutta.main(**params, if_plot=False)
    nx, nz, nvx, nvz, nt = numerical.main(**params, if_plot=False)  # Reference

    # Convert to numpy arrays
    ex, ez, evx, evz = map(np.array, [ex, ez, evx, evz])
    rx, rz, rvx, rvz = map(np.array, [rx, rz, rvx, rvz])
    nx, nz, nvx, nvz = map(np.array, [nx, nz, nvx, nvz])
    et = np.array(et)
    nt = np.array(nt)

    # Align lengths
    N = min(len(et), len(nt), len(rx))
    et = et[:N]
    ex, ez, evx, evz = ex[:N], ez[:N], evx[:N], evz[:N]
    rx, rz, rvx, rvz = rx[:N], rz[:N], rvx[:N], rvz[:N]
    nx, nz, nvx, nvz = nx[:N], nz[:N], nvx[:N], nvz[:N]

    return et, ex, ez, evx, evz, rx, rz, rvx, rvz, nx, nz, nvx, nvz


def main(x0=10, z0=5, vx0=70, vz0=30, u_val=3, m_val=1, g_val=9.8, delta=1e-3, t_final=10000):
    et, ex, ez, evx, evz, rx, rz, rvx, rvz, nx, nz, nvx, nvz = run_simulation(
        x0, z0, vx0, vz0, u_val, m_val, g_val, delta, t_final
    )

    # Plot 1: Trajectory
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ex, ez, label="Euler", color='red')
    plt.plot(rx, rz, label="RK4", color='blue')
    plt.plot(nx, nz, label="Numerical", color='green')
    plt.title("Trajectory (x vs z)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.legend()

    # Plot 2: Velocities
    plt.subplot(1, 2, 2)
    plt.plot(et, evx, 'r', label="Euler vx")
    plt.plot(et, rvx, 'b', label="RK4 vx")
    plt.plot(et, nvx, 'g', label="Numerical vx")
    plt.plot(et, evz, 'r--', label="Euler vz")
    plt.plot(et, rvz, 'b--', label="RK4 vz")
    plt.plot(et, nvz, 'g--', label="Numerical vz")
    plt.title("Velocities vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: Absolute Errors
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(et, np.abs(ex - nx), label="Euler", color='red')
    plt.plot(et, np.abs(rx - nx), label="RK4", color='blue')
    plt.title("Absolute Error in x(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(et, np.abs(ez - nz), label="Euler", color='red')
    plt.plot(et, np.abs(rz - nz), label="RK4", color='blue')
    plt.title("Absolute Error in z(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(et, np.abs(evx - nvx), label="Euler", color='red')
    plt.plot(et, np.abs(rvx - nvx), label="RK4", color='blue')
    plt.title("Absolute Error in vx(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(et, np.abs(evz - nvz), label="Euler", color='red')
    plt.plot(et, np.abs(rvz - nvz), label="RK4", color='blue')
    plt.title("Absolute Error in vz(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Statistics
    def compute_stats(label, true_vals, approx_vals):
        error = np.abs(np.array(true_vals) - np.array(approx_vals))
        return {
            'label': label,
            'max': np.max(error),
            'mean': np.mean(error),
            'final': error[-1]
        }

    stats = [
        compute_stats("Euler x", nx, ex),
        compute_stats("RK4   x", nx, rx),
        compute_stats("Euler z", nz, ez),
        compute_stats("RK4   z", nz, rz),
        compute_stats("Euler vx", nvx, evx),
        compute_stats("RK4   vx", nvx, rvx),
        compute_stats("Euler vz", nvz, evz),
        compute_stats("RK4   vz", nvz, rvz),
    ]

    print("\nComparison Statistics (Euler vs RK4 vs Numerical):")
    for s in stats:
        print(f"{s['label']:<10} | Max: {s['max']:.4e} | Mean: {s['mean']:.4e} | Final: {s['final']:.4e}")

    # Decide best method by mean error
    euler_mean = np.mean([s['mean'] for s in stats if "Euler" in s['label']])
    rk4_mean = np.mean([s['mean'] for s in stats if "RK4" in s['label']])

    print("\nAverage Mean Error:")
    print(f"Euler: {euler_mean:.4e}")
    print(f"RK4:   {rk4_mean:.4e}")

    print("\nBest method overall:", "RK4" if rk4_mean < euler_mean else "Euler")

if __name__ == "__main__":
    main()