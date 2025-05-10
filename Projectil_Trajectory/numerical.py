import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def projectile_with_drag(t, y, u, m, g):
    x, vx, z, vz = y
    sign_vx = np.sign(vx)
    sign_vz = np.sign(vz)
    dxdt = vx
    dvxdt = -(u / m) * vx**2 * sign_vx
    dzdt = vz
    dvzdt = -g - (u / m) * vz**2 * sign_vz
    return [dxdt, dvxdt, dzdt, dvzdt]

def hit_ground(t, y, *args):
    return y[2]
hit_ground.terminal = True
hit_ground.direction = -1

def main(x0=10, z0=5, vx0=70, vz0=30, u_val=3, m_val=1, g_val=9.8, delta=1e-3, t_final=10000, if_plot=True):
    y0 = [x0, vx0, z0, vz0]
    t_span = (0, t_final * delta)

    sol = solve_ivp(
        projectile_with_drag, t_span, y0, args=(u_val, m_val, g_val),
        events=hit_ground, dense_output=True, max_step=delta
    )

    t = sol.t
    x = sol.y[0]
    vx = sol.y[1]
    z = sol.y[2]
    vz = sol.y[3]

    if if_plot:
        plt.subplot(1, 2, 1)
        plt.plot(x, z, 'r')
        plt.title("Numerical Trajectory")
        plt.xlabel("x")
        plt.ylabel("z")

        plt.subplot(1, 2, 2)
        plt.plot(t, vx, 'k', label='vx')
        plt.plot(t, vz, 'b', label='vz')
        plt.title("Velocities")
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return x.tolist(), z.tolist(), vx.tolist(), vz.tolist(), t.tolist()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 11:
        args = list(map(float, sys.argv[1:]))
        result = main(*args)
    else:
        result = main()
    print("Final position (x, z):", result[0][-1], result[1][-1])
    print("Final velocity (vx, vz):", result[2][-1], result[3][-1])
