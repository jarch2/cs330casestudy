import csv
from queue import Queue
import heapq
from datetime import datetime, timedelta
import json
import random
from collections import deque

class NotUber:
    def __init__(self):
        #should be the same for every task
        self.passenger_queue = deque()
        self.load_passengers('passengers.csv')

        self.driver_queue = deque(self.load_drivers('drivers.csv'))

        self.nodes = self.load_json('node_data.json')
        self.adjacency = self.create_edges_dict('edges.csv')
        
        self.timestamp = 0
        self.driver_re_entry = []
        heapq.heapify(self.driver_re_entry)

        #specific to t1
        self.waiting_passengers = deque()
        self.waiting_drivers = deque()

    #region utils
    def create_edges_dict(self, filename):
        data = {}
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            weekday_labels = next(reader)[3:]
            for row in reader:
                for row in reader:
                    source, destination = row[0], row[1]
                    length = float(row[2])

                    if source not in data:
                        data[source] = {}
                    if destination not in data[source]:
                        data[source][destination] = {}

                    for i, label in enumerate(weekday_labels):
                        data[source][destination][label] = length / float(row[i + 3])

        return data

    def load_passengers(self, passengers_csv):
        def read_csv(filename):
            data = []
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    correct_datatype_row = []
                    # datetime
                    correct_datatype_row.append(self.convert_string_to_datetime(row[0]))
                    # source lat
                    correct_datatype_row.append(float(row[1]))
                    # source long
                    correct_datatype_row.append(float(row[2]))
                    # dest lat
                    correct_datatype_row.append(float(row[3]))
                    # dest long
                    correct_datatype_row.append(float(row[4]))
                    data.append(correct_datatype_row)
            return data
        passengers = read_csv(passengers_csv)
        for passenger in passengers:
            self.passenger_queue.append(passenger)

    def load_drivers(self, drivers_file):
        def read_csv(filename):
            data = []
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    correct_datatype_row = []
                    # datetime
                    correct_datatype_row.append(self.convert_string_to_datetime(row[0]))
                    # source lat
                    correct_datatype_row.append(float(row[1]))
                    # source long
                    correct_datatype_row.append(float(row[2]))
                    data.append(correct_datatype_row)
            return data
        drivers = read_csv(drivers_file)
        return drivers
    
    def convert_string_to_datetime(self, date_string):
        date_format = "%m/%d/%Y %H:%M:%S"
        date_object = datetime.strptime(date_string, date_format)
        return date_object

    def load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return dlat**2 + dlon**2

    #endregion
        
    def update_timestamp(self):
        passenger_time = datetime(9999, 1, 1)
        if self.passenger_queue : passenger_time = self.passenger_queue[0][0]
        driver_time = datetime(9999, 1, 1)
        if self.driver_queue : driver_time = self.driver_queue[0][0]
        re_entry_time = datetime(9999, 1, 1)
        if self.driver_re_entry : re_entry_time = self.driver_re_entry[0][0]


        self.timestamp = min(passenger_time, driver_time, re_entry_time)
        if (self.timestamp == passenger_time):
            passenger = self.passenger_queue.popleft()
            self.waiting_passengers.append(passenger)

        if (self.timestamp == driver_time):
            driver = self.driver_queue.popleft()
            self.waiting_drivers.append(driver)

        if (self.timestamp == re_entry_time):
            driver = heapq.heappop(self.driver_re_entry)
            self.waiting_drivers.append(driver)

        return self.try_match()
    
    def try_match(self):
        if self.waiting_passengers and self.waiting_drivers:
            passenger = self.waiting_passengers.popleft()
            driver = self.waiting_drivers.popleft()
            return self.assign_ride(driver, passenger)
        # no match available
        return None, None

    def assign_ride(self, driver, passenger):
        driver_node = self.find_closest_node(driver[1], driver[2])
        passenger_source_node = self.find_closest_node(passenger[1], passenger[2])
        passenger_dest_node = self.find_closest_node(passenger[3], passenger[4])

        # float representing hours: driver location to passenger pick up time
        travel_time_driver_to_passenger = self.calc_travel_time(driver_node, passenger_source_node, self.timestamp)

        # float representing hours: pick up to drop off time
        travel_time = self.calc_travel_time(passenger_source_node, passenger_dest_node, self.timestamp)

        # float representing hours
        total_travel_time = travel_time_driver_to_passenger + travel_time

        # date time object
        arrival_time = self.timestamp + timedelta(hours=total_travel_time)

        # drivers log off starting after 1 hour of being logged on
        # drivers always log off after finsihing a ride if they've been logged on for more than 4 hours
        hours_logged = (self.timestamp - driver[0]).total_seconds() / 3600
        if random.random() > ((hours_logged - 1) / 3):
            # set the new entry time, lat, and lon for driver as the drop off of the previous passenger
            driver[0] = arrival_time
            driver[1] = self.nodes[passenger_dest_node]['lat']
            driver[2] = self.nodes[passenger_dest_node]['lon']
            heapq.heappush(self.driver_re_entry, driver)

        # for benchmarking
        d1 = (arrival_time - passenger[0]).total_seconds() / 60
        d2 = (travel_time - travel_time_driver_to_passenger) * 60 
        return d1, d2

    def find_closest_node(self, lat, lon):
        closest_node = None
        min_distance = float('inf')
        for node_id, coords in self.nodes.items():
            dist = self.euclidean_distance(lat, lon, coords['lat'], coords['lon'])
            if dist < min_distance:
                min_distance = dist
                closest_node = node_id
        return closest_node
    
    def dijkstra(self, source, dest, day_hour):
        pq = [(0, source)]  # Priority queue as a min-heap with (distance, node)
        visited = set()

        while pq:
            (dist, current_node) = heapq.heappop(pq)

            if current_node == dest:
                return dist

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self.adjacency.get(current_node, {}):
                if neighbor not in visited:
                    weight = self.adjacency[current_node][neighbor][day_hour]  
                    heapq.heappush(pq, (dist + weight, neighbor))

        print("xxxyyyzzz")
        return float('inf')


    def calc_travel_time(self, source_node, dest_node, current_time):
        def datetime_to_string(dt):
            if dt.weekday() >= 5:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            hour = dt.hour
            return f"{day_type}_{hour}"
        
        return self.dijkstra(source_node, dest_node, datetime_to_string(current_time))



# experiment

start_time = datetime.now()
print("starting at ", start_time)

not_uber = NotUber()

preprocess_time = datetime.now()
print("preprocessing done in " + str((preprocess_time - start_time).total_seconds() / 60))


d1_total = 0
d2_total = 0

num_passengers = len(not_uber.passenger_queue)
num_drivers = len(not_uber.driver_queue)

while not_uber.passenger_queue or not_uber.driver_queue or not_uber.driver_re_entry:
    d1, d2 = not_uber.update_timestamp()
    if d1 : d1_total += d1
    if d2 : d2_total += d2


end_time = datetime.now()
# duration in minutes
duration = (end_time - start_time).total_seconds() / 60
sim_duration = (end_time - preprocess_time).total_seconds() / 60

unmatched_passengers = len(not_uber.waiting_passengers)
unmatched_drivers = len(not_uber.waiting_drivers)

print("unmatched passengers: ", unmatched_passengers)
print("unmatched drivers: ", unmatched_drivers)

print("average d1: ", d1_total / (num_passengers - unmatched_passengers))
print("average d2: ", d2_total / num_drivers)

print("total time: ", duration)
print("simulation time: ", sim_duration)
