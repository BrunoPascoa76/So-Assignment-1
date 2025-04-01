# the "unit of time" will be minutes (since I prefer base units to be integers)

import argparse
import random as rng
import simpy
import logging
from logging import debug, info
from util import positive_int,non_negative_int, probability


logging.basicConfig(
    filename="log.txt",
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s",
    filemode="w"
)

MEAN_INTERARRIVAL_TIME = 2*60
NUM_REPAIR_STATIONS = 2
NUM_INSPECTION_STATIONS = 1
MIN_INSPECTION_TIME = 15
MAX_INSPECTION_TIME = 1.05*60
REPAIR_PROBABILITY = 0.30
MIN_REPAIR_TIME = 2.1*60
MAX_REPAIR_TIME = 4.5*60
RANDOM_SEED = 107418
SIM_TIME=160*60

def bus(env, id, inspection_stations, repair_stations):
    arrival_time=env.now

    with inspection_stations.request() as req:
        yield req # wait in the inspection queue (if needed)
        exit_queue_time=env.now
        debug(f"{env.now}: Bus {id} starts inspection after waiting for {exit_queue_time-arrival_time:.2f} minutes")

        yield env.timeout(rng.uniform(MIN_INSPECTION_TIME,MAX_INSPECTION_TIME)) # simulate inspection time
        leave_inspection_time=env.now
        debug(f"{env.now}: Bus {id} finishes inspection (took {leave_inspection_time-exit_queue_time:.2f} minutes)")

    if rng.random() <= REPAIR_PROBABILITY: # decide whether or not it needs repair
        debug(f"{env.now}: Bus {id} needs repair")

        with repair_stations.request() as req:
            repair_arrival_time=env.now

            yield req
            exit_queue_time=env.now
            debug(f"{env.now}: Bus {id} starts repair after waiting for {exit_queue_time-repair_arrival_time:.2f} minutes")
            
            yield env.timeout(rng.uniform(MIN_REPAIR_TIME,MAX_REPAIR_TIME)) # simulate repair time
            leave_repair_time=env.now
            debug(f"{env.now}: Bus {id} finishes repair and leaves (took {leave_repair_time-exit_queue_time:.2f} minutes)")

    else:
        debug(f"{env.now}: Bus {id} does not need repair and leaves")

def bus_generator(env,inspection_stations,repair_stations):
    """Generates new buses at exponentially distributed intervals"""
    bus_id=0
    while True:
        pre_wait_time=env.now
        yield env.timeout(rng.expovariate(1/MEAN_INTERARRIVAL_TIME)) # wait for next bus

        debug(f"{env.now}: Waited {env.now-pre_wait_time:.2f} minutes for next bus")

        bus_id+=1
        debug(f"{env.now}: Bus {bus_id} arrives for inspection")
        env.process(bus(env,bus_id,inspection_stations,repair_stations))


def simulate():
    rng.seed(RANDOM_SEED)
    env=simpy.Environment()

    parameters={
        "mean_interarrival_time": MEAN_INTERARRIVAL_TIME,
        "num_repair_stations": NUM_REPAIR_STATIONS,
        "num_inspection_stations": NUM_INSPECTION_STATIONS,
        "min_inspection_time": MIN_INSPECTION_TIME,
        "max_inspection_time": MAX_INSPECTION_TIME,
        "repair_probability": REPAIR_PROBABILITY,
        "min_repair_time": MIN_REPAIR_TIME,
        "max_repair_time": MAX_REPAIR_TIME,
        "random_seed": RANDOM_SEED,
        "sim_time": SIM_TIME
    }

    inspection_stations=simpy.Resource(env, capacity=NUM_INSPECTION_STATIONS)
    repair_stations=simpy.Resource(env, capacity=NUM_REPAIR_STATIONS)

    env.process(bus_generator(env,inspection_stations,repair_stations))

    debug(f"{0}: Starting simulation with parameters {parameters}")
    env.run(until=SIM_TIME)


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

    args=parser.parse_args()

    MEAN_INTERARRIVAL_TIME=args.arrival_time
    NUM_REPAIR_STATIONS=args.num_repair_stations
    NUM_INSPECTION_STATIONS=args.num_inspection_stations
    MIN_INSPECTION_TIME=args.min_inspection_time
    MAX_INSPECTION_TIME=args.max_inspection_time
    REPAIR_PROBABILITY=args.repair_probability
    MIN_REPAIR_TIME=args.min_repair_time
    MAX_REPAIR_TIME=args.max_repair_time
    

    if(MIN_INSPECTION_TIME > MAX_INSPECTION_TIME):
        parser.error("--min_inspection_time must be less than --max_inspection_time")
    
    if(MIN_REPAIR_TIME > MAX_REPAIR_TIME):
        parser.error("--min_repair_time must be less than --max_repair_time")

    simulate()

