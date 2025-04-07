import matplotlib.pyplot as plt
from labellines import labelLines
import numpy as np
import sys

def initialize(x0, vx0, z0, vz0):
    global x, vx, z, vz
    global result_x, result_vx, result_z, result_vz, t_span

    x, vx = x0, vx0
    z, vz = z0, vz0

    result_x = [x]
    result_vx = [vx]
    result_z = [z]
    result_vz = [vz]
    t_span = [0]

def observe(t):
    global x, vx, z, vz
    global result_x, result_vx, result_z, result_vz, t_span

    result_x.append(x)
    result_vx.append(vx)
    result_z.append(z)
    result_vz.append(vz)
    t_span.append(t)

def update():
    global x, vx, z, vz
    global g, u, m, delta_t

    dx_dt = lambda v: v
    dvx_dt = lambda v: (-u * (v**2) * np.sign(v)) / m

    dz_dt = lambda v: v
    dvz_dt = lambda v: (-m * g - u * (v**2) * np.sign(v)) / m

    # RK4 for x and vx
    K1x = delta_t * dx_dt(vx)
    K1vx = delta_t * dvx_dt(vx)

    K2x = delta_t * dx_dt(vx + K1vx/2)
    K2vx = delta_t * dvx_dt(vx + K1vx/2)

    K3x = delta_t * dx_dt(vx + K2vx/2)
    K3vx = delta_t * dvx_dt(vx + K2vx/2)

    K4x = delta_t * dx_dt(vx + K3vx)
    K4vx = delta_t * dvx_dt(vx + K3vx)

    x += (K1x + 2*K2x + 2*K3x + K4x) / 6
    vx += (K1vx + 2*K2vx + 2*K3vx + K4vx) / 6

    # RK4 for z and vz
    K1z = delta_t * dz_dt(vz)
    K1vz = delta_t * dvz_dt(vz)

    K2z = delta_t * dz_dt(vz + K1vz/2)
    K2vz = delta_t * dvz_dt(vz + K1vz/2)

    K3z = delta_t * dz_dt(vz + K2vz/2)
    K3vz = delta_t * dvz_dt(vz + K2vz/2)

    K4z = delta_t * dz_dt(vz + K3vz)
    K4vz = delta_t * dvz_dt(vz + K3vz)

    z += (K1z + 2*K2z + 2*K3z + K4z) / 6
    vz += (K1vz + 2*K2vz + 2*K3vz + K4vz) / 6

def main(x0=10, z0=5, vx0=70, vz0=30, u_val=3, m_val=1, g_val=9.8, delta=0.001, t_final=4000, if_plot=True):
    global g, u, m, delta_t

    g = g_val
    u = u_val
    m = m_val
    delta_t = delta

    initialize(x0, vx0, z0, vz0)

    for t in range(t_final):
        update()
        if z < 0:
            break
        observe(t * delta_t)

    if if_plot:
        plt.subplot(1,2,1)
        plt.plot(result_x, result_z, 'r')
        plt.title('Trajectory')
        plt.xlabel('x')
        plt.ylabel('z')

        plt.subplot(1,2,2)
        plt.plot(t_span, result_vx, 'k', label='vx')
        plt.plot(t_span, result_vz, 'b', label='vz')
        plt.title('Velocities')
        plt.xlabel('Time')
        plt.ylabel('units/(time^2)')
        plt.legend()
        plt.show()

    return result_x, result_z, result_vx, result_vz, t_span

if __name__ == '__main__':
    if len(sys.argv) == 11:
        args = list(map(float, sys.argv[1:]))
        result = main(*args)
    else:
        result = main()

    print("Final position (x, z):", result[0][-1], result[1][-1])
    print("Final velocity (vx, vz):", result[2][-1], result[3][-1])
