# the "unit of time" will be minutes (since I prefer base units to be integers)

import random as rng
import simpy
import logging
from logging import debug, info


logging.basicConfig(
    filename="log.txt",
    level=logging.DEBUG,
    format="%(message)s"
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
        debug(f"{env.now}: Bus {id} starts inspection after waiting for {exit_queue_time-arrival_time} minutes")

        yield env.timeout(rng.uniform(MIN_INSPECTION_TIME,MAX_INSPECTION_TIME)) # simulate inspection time
        leave_inspection_time=env.now
        debug(f"{env.now}: Bus {id} finishes inspection (took {leave_inspection_time-exit_queue_time} minutes)")

    if rng.random() <= REPAIR_PROBABILITY: # decide whether or not it needs repair
        debug(f"{env.now}: Bus {id} needs repair")

        with repair_stations.request() as req:
            repair_arrival_time=env.now

            yield req
            exit_queue_time=env.now
            debug(f"{env.now}: Bus {id} starts repair after waiting for {exit_queue_time-repair_arrival_time} minutes")
            
            yield env.timeout(rng.uniform(MIN_REPAIR_TIME,MAX_REPAIR_TIME)) # simulate repair time
            leave_repair_time=env.now
            debug(f"{env.now}: Bus {id} finishes repair and leaves (took {leave_repair_time-exit_queue_time} minutes)")

    else:
        debug(f"{env.now}: Bus {id} does not need repair and leaves")

def bus_generator(env,inspection_stations,repair_stations):
    """Generates new buses at exponentially distributed intervals"""
    bus_id=0
    while True:
        pre_wait_time=env.now
        yield env.timeout(rng.expovariate(1/MEAN_INTERARRIVAL_TIME)) # wait for next bus

        bus_id+=1
        debug(f"{env.now}: Bus {bus_id} arrives for inspection (waited {env.now-pre_wait_time} minutes)")
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


