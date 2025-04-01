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
    
    global g,u,m, delta_t
    
    next_x = x + delta_t * vx
    next_vx = vx + delta_t * (-(u/m)*(vx**2)*np.sign(vx))
    
    next_z = z + delta_t * vz
    next_vz = vz + delta_t * (-g -(u/m)*(vz**2)*np.sign(vz))
    
    x, vx = next_x, next_vx
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
    

if __name__ == main():
    # Need to add to change values in terminal or when using file in another file
    main()