import matplotlib.pyplot as plt
from labellines import labelLines
import numpy as np
import sys

def initialize():
    global x, vx
    global result_x, result_vx
    global z, vz
    global result_z, result_vz
    
    global t_span
    
    x = 10 # Try various x's
    vx = 70 # Try various vx's
    
    z = 5 # Try various z's
    vz = 30 # Try various vz's
    
    result_x = [x]
    result_vx = [vx]
    
    result_z = [z]
    result_vz = [vz]
    
    t_span.append(0) # Time interval
    
def observe():
    global x, vx
    global result_x, result_vx
    global z, vz
    global result_z, result_vz
    
    global t_span
    
    result_x.append(x)
    result_vx.append(vx)
    
    result_z.append(z)
    result_vz.append(vz)
    
    t_span.append(delta_t)
    
def update():
    global x, vx
    global result_x, result_vx
    global z, vz
    global result_z, result_vz

    global g, u, m, delta_t

    dx_dt = lambda current_vx: current_vx
    dvx_dt = lambda current_vx: (-u * (current_vx**2) * np.sign(current_vx)) / m

    dz_dt = lambda current_vz: current_vz
    dvz_dt = lambda current_vz: (-m * g - u * (current_vz**2) * np.sign(current_vz)) / m

    K1x = delta_t * dx_dt(vx)
    K1vx = delta_t * dvx_dt(vx)

    K2x = delta_t * dx_dt(vx + K1vx/2)
    K2vx = delta_t * dvx_dt(vx + K1vx/2)

    K3x = delta_t * dx_dt(vx + K2vx/2)
    K3vx = delta_t * dvx_dt(vx + K2vx/2)

    K4x = delta_t * dx_dt(vx + K3vx)
    K4vx = delta_t * dvx_dt(vx + K3vx)

    next_x = x + (K1x + 2 * K2x + 2 * K3x + K4x) / 6
    next_vx = vx + (K1vx + 2 * K2vx + 2 * K3vx + K4vx) / 6

    x, vx = next_x, next_vx

    K1z = delta_t * dz_dt(vz)
    K1vz = delta_t * dvz_dt(vz)

    K2z = delta_t * dz_dt(vz + K1vz/2)
    K2vz = delta_t * dvz_dt(vz + K1vz/2)

    K3z = delta_t * dz_dt(vz + K2vz/2)
    K3vz = delta_t * dvz_dt(vz + K2vz/2)

    K4z = delta_t * dz_dt(vz + K3vz)
    K4vz = delta_t * dvz_dt(vz + K3vz)

    next_z = z + (K1z + 2 * K2z + 2 * K3z + K4z) / 6
    next_vz = vz + (K1vz + 2 * K2vz + 2 * K3vz + K4vz) / 6

    z, vz = next_z, next_vz
    
def main():
    global g, u, m, delta_t, t_final, t_span
    
    g = 9.8
    u = 3 # Try various ranges
    m = 1 # Try various ranges
    
    t_final = 4*10 ** 3 # Try various ranges
    
    delta_t = 0.001 # Try various ranges
    
    t_span = []
    
    initialize()
    
    for t in range(t_final):
        update()
        observe()
        if z < -0:
            break
    
    plt.subplot(1,2,1)
    plt.plot(result_x, result_z, 'r')
    plt.title('Trajectory')
    plt.xlabel('x')
    plt.ylabel('z')
    
    plt.subplot(1,2,2)
    plt.plot(range(len(result_vx)), result_vx, 'k', label='vx')
    plt.plot(range(len(result_vz)), result_vz, 'b', label='vz')

    plt.title('Velocities')
    plt.xlabel('Iterations')
    plt.ylabel('units/(time^2)')
    plt.legend()
    plt.show()
    
    
main()