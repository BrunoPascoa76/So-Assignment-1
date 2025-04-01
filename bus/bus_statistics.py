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

    def update_inspection_utilization(self):
        pass

    def update_repair_utilization(self):
        pass

    def update_inspection_queue_size(self):
        pass

    def update_repair_queue_size(self):
        pass

    def update_inspection_delay(self):
        pass

    def update_repair_delay(self):
        pass

    @property
    def mean_inspection_utilization(self):
        pass

    @property
    def mean_repair_utilization(self):
        pass

    @property
    def mean_inspection_queue_size(self):
        pass

    @property
    def mean_repair_queue_size(self):
        pass

    @property
    def mean_inspection_delay(self):
        pass

    @property
    def mean_repair_delay(self):
        pass
