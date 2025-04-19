class Statistics:
    def __init__(self,env,inspection_stations,repair_stations):
        self.env=env
        self.inspection_stations=inspection_stations
        self.repair_stations=repair_stations

        self.previous_inspection_change_time=env.now
        self.previous_repair_change_time=env.now

        self.inspection_queue_sizes=[]
        self.inspection_delays=[]
        self.inspection_utilizations=[]

        self.repair_utilizations=[]
        self.repair_queue_sizes=[]
        self.repair_delays=[]

        self.durations=[] # time each bus spent in the system, to calculate average exit rate

    def update_inspection_utilization(self,current_value=None):
        change_time=self.env.now
        elapsed_time=change_time-self.previous_inspection_change_time
        current_utilization=current_value if current_value is not None else self.inspection_stations.count

        self.inspection_utilizations.append((current_utilization*elapsed_time,current_utilization,change_time)) #last 2 are for plotting
        self.previous_inspection_change_time=change_time
    
    def update_repair_utilization(self,current_value=None):
        change_time=self.env.now
        elapsed_time=change_time-self.previous_repair_change_time
        current_utilization=current_value if current_value is not None else self.repair_stations.count

        self.repair_utilizations.append((current_utilization*elapsed_time,current_utilization,change_time)) #last 2 are for plotting
        self.previous_repair_change_time=change_time


    

    def update_inspection_queue_size(self,current_value=None):
        change_time=self.env.now
        elapsed_time=change_time-self.previous_inspection_change_time
        current_queue_size=current_value if current_value is not None else len(self.inspection_stations.queue)

        self.inspection_queue_sizes.append((current_queue_size*elapsed_time,current_queue_size,change_time)) #last 2 are for plotting

    def update_repair_queue_size(self,current_value=None):
        change_time=self.env.now
        elapsed_time=change_time-self.previous_repair_change_time
        current_queue_size=current_value if current_value is not None else len(self.repair_stations.queue)

        self.repair_queue_sizes.append((current_queue_size*elapsed_time,current_queue_size,change_time)) #last 2 are for plotting

    def update_inspection_delay(self,delay):
        self.inspection_delays.append((delay,self.env.now))

    def update_repair_delay(self,delay):
        self.repair_delays.append((delay,self.env.now))

    def add_duration(self,duration):
        self.durations.append((duration,self.env.now))


    @property
    def mean_inspection_utilization(self):
        weighted_utilizations=[util for (util,_,_) in self.inspection_utilizations]
        return sum(weighted_utilizations)/self.env.now/self.inspection_stations.capacity

    @property
    def mean_repair_utilization(self):
        weighted_utilizations=[util for (util,_,_) in self.repair_utilizations]
        return sum(weighted_utilizations)/self.env.now/self.repair_stations.capacity

    @property
    def mean_inspection_queue_size(self):
        weighted_sizes=[size for (size,_,_) in self.inspection_queue_sizes]
        return sum(weighted_sizes)/self.env.now

    @property
    def mean_repair_queue_size(self):
        weighted_sizes=[size for (size,_,_) in self.repair_queue_sizes]
        return sum(weighted_sizes)/self.env.now

    @property
    def mean_inspection_delay(self):
        delays=[delay for (delay,_) in self.inspection_delays]
        return sum(delays)/max(len(delays),1)

    @property
    def mean_repair_delay(self):
        delays=[delay for (delay,_) in self.repair_delays]
        return sum(delays)/max(len(delays),1)
    
    @property
    def mean_duration(self):
        durations=[duration for (duration,_) in self.durations]
        return sum(durations)/max(len(durations),1)
