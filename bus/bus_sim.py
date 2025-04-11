# the "unit of time" will be minutes (since I prefer base units to be integers)

import argparse
import random as rng
import simpy
import logging
from logging import debug, info
from bus_statistics import Statistics
from util import positive_int,non_negative_int, probability,run_data,init_plot,update_plot_double,update_plot_triple,fig,ax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


logging.basicConfig(
    filename="log.txt",
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s",
    filemode="w"
)

all_results=[]

MEAN_INTERARRIVAL_TIME = 2*60
NUM_REPAIR_STATIONS = 2
NUM_INSPECTION_STATIONS = 1
MIN_INSPECTION_TIME = 15
MAX_INSPECTION_TIME = 1.05*60
REPAIR_PROBABILITY = 0.30
MIN_REPAIR_TIME = 2.1*60
MAX_REPAIR_TIME = 4.5*60
SIM_TIME=160*60
RUNS=1

def bus(env, id, inspection_stations, repair_stations,statistics):
    arrival_time=env.now


    statistics.update_inspection_queue_size() # before (potential) increase due to request
    with inspection_stations.request() as req: 
        
        statistics.update_inspection_queue_size() #before decrease from exiting yield
        statistics.update_inspection_utilization() #before increase from leaving queue
        yield req # wait in the inspection queue (if needed)

        exit_queue_time=env.now
        statistics.update_inspection_delay(exit_queue_time-arrival_time)
        info(f"{env.now}: Bus {id} starts inspection after waiting for {exit_queue_time-arrival_time:.2f} minutes")

        yield env.timeout(rng.uniform(MIN_INSPECTION_TIME,MAX_INSPECTION_TIME)) # simulate inspection time
        leave_inspection_time=env.now
        info(f"{env.now}: Bus {id} finishes inspection (took {leave_inspection_time-exit_queue_time:.2f} minutes)")

        statistics.update_inspection_utilization() #before decrease due to releasing resource




    if rng.random() <= REPAIR_PROBABILITY: # decide whether or not it needs repair
        info(f"{env.now}: Bus {id} needs repair")
        
        statistics.update_repair_queue_size()
        with repair_stations.request() as req:
            repair_arrival_time=env.now

            statistics.update_repair_queue_size()
            statistics.update_repair_utilization()
            yield req

            exit_queue_time=env.now
            statistics.update_repair_delay(exit_queue_time-repair_arrival_time)
            info(f"{env.now}: Bus {id} starts repair after waiting for {exit_queue_time-repair_arrival_time:.2f} minutes")
            
            yield env.timeout(rng.uniform(MIN_REPAIR_TIME,MAX_REPAIR_TIME)) # simulate repair time
            leave_repair_time=env.now
            info(f"{env.now}: Bus {id} finishes repair and leaves (took {leave_repair_time-exit_queue_time:.2f} minutes)")

            statistics.update_repair_utilization()
        statistics.add_duration(env.now-arrival_time)

    else:
        statistics.add_duration(env.now-arrival_time)
        info(f"{env.now}: Bus {id} does not need repair and leaves")

def bus_generator(env,inspection_stations,repair_stations,statistics):
    """Generates new buses at exponentially distributed intervals"""
    bus_id=0
    while True:
        pre_wait_time=env.now
        yield env.timeout(rng.expovariate(1/MEAN_INTERARRIVAL_TIME)) # wait for next bus

        info(f"{env.now}: Waited {env.now-pre_wait_time:.2f} minutes for next bus")

        bus_id+=1
        info(f"{env.now}: Bus {bus_id} arrives for inspection")
        env.process(bus(env,bus_id,inspection_stations,repair_stations,statistics))


def simulate():
    env=simpy.Environment()

    inspection_stations=simpy.Resource(env, capacity=NUM_INSPECTION_STATIONS)
    repair_stations=simpy.Resource(env, capacity=NUM_REPAIR_STATIONS)

    statistics=Statistics(env,inspection_stations,repair_stations)

    env.process(bus_generator(env,inspection_stations,repair_stations,statistics))
    env.run(until=SIM_TIME)

    return statistics


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Simulates a bus inspection (and repair) system")
    
    parser.add_argument("--arrival_time",type=positive_int,default=MEAN_INTERARRIVAL_TIME,help="Mean interarrival time for buses (in minutes)")

    parser.add_argument("--min_inspection_time",type=non_negative_int,default=MIN_INSPECTION_TIME,help="Minimum inspection time (in minutes)")
    parser.add_argument("--max_inspection_time",type=non_negative_int,default=MAX_INSPECTION_TIME,help="Maximum inspection time (in minutes)")

    parser.add_argument("--min_repair_time",type=non_negative_int,default=MIN_REPAIR_TIME,help="Minimum repair time (in minutes)")
    parser.add_argument("--max_repair_time",type=non_negative_int,default=MAX_REPAIR_TIME,help="Maximum repair time (in minutes)")


    parser.add_argument("--num_inspection_stations",type=positive_int,default=NUM_INSPECTION_STATIONS,help="Number of inspection stations")
    parser.add_argument("--num_repair_stations",type=positive_int,default=NUM_REPAIR_STATIONS,help="Number of repair stations")
    parser.add_argument("--repair_probability",type=probability,default=REPAIR_PROBABILITY,help="Probability of a bus needing repair")

    parser.add_argument("--time",type=positive_int,default=SIM_TIME,help="Simulation time (in minutes)")

    parser.add_argument("--runs",type=positive_int,default=RUNS,help="Number of runs to simulate")

    parser.add_argument("--plot",action="store_true",default=False,help="Plot results")

    args=parser.parse_args()

    MEAN_INTERARRIVAL_TIME=args.arrival_time
    NUM_REPAIR_STATIONS=args.num_repair_stations
    NUM_INSPECTION_STATIONS=args.num_inspection_stations
    MIN_INSPECTION_TIME=args.min_inspection_time
    MAX_INSPECTION_TIME=args.max_inspection_time
    REPAIR_PROBABILITY=args.repair_probability
    MIN_REPAIR_TIME=args.min_repair_time
    MAX_REPAIR_TIME=args.max_repair_time
    SIM_TIME=args.time    
    RUNS=args.runs

    if(MIN_INSPECTION_TIME > MAX_INSPECTION_TIME):
        parser.error("--min_inspection_time must be less than --max_inspection_time")
    
    if(MIN_REPAIR_TIME > MAX_REPAIR_TIME):
        parser.error("--min_repair_time must be less than --max_repair_time")

    parameters={
        "mean_interarrival_time": MEAN_INTERARRIVAL_TIME,
        "num_repair_stations": NUM_REPAIR_STATIONS,
        "num_inspection_stations": NUM_INSPECTION_STATIONS,
        "min_inspection_time": MIN_INSPECTION_TIME,
        "max_inspection_time": MAX_INSPECTION_TIME,
        "repair_probability": REPAIR_PROBABILITY,
        "min_repair_time": MIN_REPAIR_TIME,
        "max_repair_time": MAX_REPAIR_TIME,
        "sim_time": SIM_TIME,
        "runs": RUNS
    }
    info(f"{0}: Starting simulation with parameters {parameters}")

    separator_string="\n"+"="*40+" Run {:3d} "+"="*40+"\nINFO-"

    inspection_queue_size_sum=0
    inspection_utilization_sum=0
    repair_queue_size_sum=0
    repair_utilization_sum=0
    inspection_delay_sum=0
    repair_delay_sum=0
    duration_sum=0

    print("starting simulation...")

    for i in range(RUNS):
        print(f"run {i+1}... ",end="")
        info(separator_string.format(i+1))
        results=simulate()
        print("DONE")

        all_results.append(results)

        info(" ")
        info(f"{results.env.now}: Finished simulation run {i+1}")
        info(f"{results.env.now}: Average inspection queue size: {results.mean_inspection_queue_size:.2f}")
        inspection_queue_size_sum+=results.mean_inspection_queue_size
        info(f"{results.env.now}: Average inspection utilization: {results.mean_inspection_utilization:.2f}")
        inspection_utilization_sum+=results.mean_inspection_utilization

        info(f"{results.env.now}: Average repair queue size: {results.mean_repair_queue_size:.2f}")
        repair_queue_size_sum+=results.mean_repair_queue_size
        info(f"{results.env.now}: Average repair utilization: {results.mean_repair_utilization:.2f}")
        repair_utilization_sum+=results.mean_repair_utilization



        info(f"{results.env.now}: Average inspection delay: {results.mean_inspection_delay:.2f}")
        inspection_delay_sum+=results.mean_inspection_delay
        info(f"{results.env.now}: Average repair delay: {results.mean_repair_delay:.2f}")
        repair_delay_sum+=results.mean_repair_delay

        info(f"{results.env.now}: Average bus duration: {results.mean_duration}")
        duration_sum+=results.mean_duration
        info(f"{results.env.now}: Average bus exit rate: {1/max(results.mean_duration,1)}")
        info(f"{results.env.now}: Average arrival rate: {1/max(MEAN_INTERARRIVAL_TIME,1)}")
        info(f"{results.env.now}: Is overloaded? {1/max(results.mean_duration,1) < 1/max(MEAN_INTERARRIVAL_TIME,1)}")

    print("calculating averages... ",end="")
    info("\n"+"="*40+" Final Results"+"="*40)

    info(f"Average inspection queue size: {inspection_queue_size_sum/RUNS:.2f}")
    info(f"Average inspection utilization: {inspection_utilization_sum/RUNS:.2f}")
    info(f"Average repair queue size: {repair_queue_size_sum/RUNS:.2f}")
    info(f"Average repair utilization: {repair_utilization_sum/RUNS:.2f}")

    info(f"Average inspection delay: {inspection_delay_sum/RUNS:.2f}")
    info(f"Average repair delay: {repair_delay_sum/RUNS:.2f}")

    info(f"Average bus duration: {duration_sum/RUNS}")
    info(f"Average bus exit rate: {1/(duration_sum/RUNS)}")
    info(f"Average arrival rate: {1/MEAN_INTERARRIVAL_TIME}")
    info(f"Is overloaded? {1/(duration_sum/RUNS) < 1/MEAN_INTERARRIVAL_TIME}")

    info("="*40+"END"+"="*40)

    print("DONE")



    if args.plot:
        print("plotting results... ",end="")
        os.makedirs("results", exist_ok=True)
        # Make gifs

        # inspection queue size
        run_data=[np.array(run.inspection_queue_sizes) for run in all_results]
        ani = FuncAnimation(fig, update_plot_triple, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"inspection queue size"))
        ani.save('results/inspection_queue_size.gif', writer='pillow', fps=1)
        plt.close()
    
        # inspection utilization
        run_data=[np.array(run.inspection_utilizations) for run in all_results]
        ani = FuncAnimation(fig, update_plot_triple, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"inspection utilization"))
        ani.save('results/inspection_utilization.gif', writer='pillow', fps=1)
        plt.close()
    
    
    
        # repair queue size
        run_data=[np.array(run.repair_queue_sizes) for run in all_results]
        ani= FuncAnimation(fig,update_plot_triple, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"repair queue size"))
        ani.save('results/repair_queue_size.gif',writer='pillow',fps=1)
    
        # repair utilization
        run_data=[np.array(run.repair_utilizations) for run in all_results]
        ani= FuncAnimation(fig,update_plot_triple, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"repair utilization"))
        ani.save('results/repair_utilization.gif',writer='pillow',fps=1)
    
        # inspection delay
        run_data=[np.array(run.inspection_delays) for run in all_results]
        ani= FuncAnimation(fig,update_plot_double, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"inspection delay (minutes)"))
        ani.save('results/inspection_delay.gif',writer='pillow',fps=1)
    
        # repair delay
        run_data=[np.array(run.repair_delays) for run in all_results]
        ani= FuncAnimation(fig,update_plot_double, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"repair delay (minutes)"))
        ani.save('results/repair_delay.gif',writer='pillow',fps=1)
    
    
        # bus duration
        run_data=[np.array(run.durations) for run in all_results]
        ani= FuncAnimation(fig,update_plot_double, frames=len(run_data), init_func=init_plot, blit=True, interval=1000,fargs=(run_data,"bus duration (minutes)"))
        ani.save('results/bus_duration.gif',writer='pillow',fps=1)
    
        print("DONE")
