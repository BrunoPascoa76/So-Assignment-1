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

    next_x = x + delta_t * vx
    next_vx = vx + delta_t * (-(u/m)*(vx**2)*np.sign(vx))

    next_z = z + delta_t * vz
    next_vz = vz + delta_t * (-g -(u/m)*(vz**2)*np.sign(vz))

    x, vx = next_x, next_vx
    z, vz = next_z, next_vz

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
        print("Final position (x, z):", result[0][-1], result[1][-1])
        print("Final velocity (vx, vz):", result[2][-1], result[3][-1])
    else:
        result = main()
        print("Final position (x, z):", result[0][-1], result[1][-1])
        print("Final velocity (vx, vz):", result[2][-1], result[3][-1])
