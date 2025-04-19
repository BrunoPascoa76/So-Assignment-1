#argument validation functions

import argparse
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
step_plot = ax.step([], [], where='pre', color='steelblue', linewidth=2)[0]

run_data=[]

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")
    return ivalue

def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} must be a non-negative integer")
    return ivalue

def probability(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"{value} must be a probability value between 0 and 1")
    return fvalue

def init_plot():
    step_plot.set_data([], [])
    return step_plot,

def update_plot_triple(frame,run_data,metric_name):
    run = run_data[frame]
    values = [item[1] for item in run]
    times = [item[2] for item in run]
    
    step_plot.set_data(times, values)
    
    # Adjust view
    ax.set_xlim(min(times)-0.5, max(times)+0.5)
    ax.set_ylim(min(values)-0.5, max(values)+0.5)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Run {frame+1}/{len(run_data)}')
    
    return step_plot,

def update_plot_double(frame,run_data,metric_name):
    run = run_data[frame]
    values = [item[0] for item in run]
    times = [item[1] for item in run]
    
    step_plot.set_data(times, values)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(metric_name)
    
    ax.set_xlim(min(times)-0.5, max(times)+0.5)
    ax.set_ylim(min(values)-0.5, max(values)+0.5)
    ax.set_title(f'Run {frame+1}/{len(run_data)}')

    return step_plot,