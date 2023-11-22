import csv
from queue import Queue
import heapq
from datetime import datetime, timedelta
import json
import random
from collections import deque

class Node:
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None

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
        self.passenger_re_entry = []
        heapq.heapify(self.driver_re_entry)
        heapq.heapify(self.passenger_re_entry)

        #specific to t3
        self.waiting_passengers = deque()
        self.waiting_drivers = deque()

        self.driver_nodes =  {}
        self.passenger_nodes = {}

        self.node_tree = self.build_kd_tree(list(self.nodes.keys()), 0)

        self.total_pickup_time = 0
        self.total_delivery_time = 0
        self.total_passenger_match_wait = 0

        self.heuristic_mph = 0
        self.passengers_left = 0

        self.sleeping_drivers = 0
        self.sleeping_passengers = 0

    def build_kd_tree(self, nodes, depth):
        if len(nodes) == 0:
            return None
        
        keys = ["lat", "lon"]
        nodes.sort(key=lambda x: self.nodes[x][keys[depth%2]])
        middle = len(nodes) // 2

        new_node = Node(nodes[middle])

        new_node.left = self.build_kd_tree(nodes[:middle], depth + 1)
        new_node.right = self.build_kd_tree(nodes[middle + 1:], depth + 1)

        return new_node
    
    def tree_nearest_node(self, lat, lon):
        best = self.node_tree.data
        best_dist = float('inf')

        def search(tree, depth):
            
            if tree is None:
                return
            
            nonlocal best
            nonlocal best_dist
            
            tree_lat = self.nodes[tree.data]['lat']
            tree_lon = self.nodes[tree.data]['lon']
            distance = self.euclidean_distance(lat, lon, tree_lat, tree_lon)
            if distance < best_dist:
                best = tree.data
                best_dist = distance
            
            diff = 0
            if depth % 2 == 0:
                diff = lat - self.nodes[tree.data]['lat']
            else:
                diff = lon - self.nodes[tree.data]['lon']

            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left
            
            search(tree=close, depth=depth+1)
            if diff**2 < best_dist:
                search(tree=away, depth=depth+1)
        
        search(self.node_tree, 0)
        return best


    #region utils
    def create_edges_dict(self, filename):
        data = {}
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            weekday_labels = next(reader)[3:]
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
        passenger_re_entry_time = datetime(9999,1,1)
        if self.passenger_re_entry : passenger_re_entry_time = self.passenger_re_entry[0][0]


        self.timestamp = min(passenger_time, driver_time, re_entry_time)

        if (self.timestamp == passenger_time):
            passenger = self.passenger_queue.popleft()
            passenger.append(passenger[0])
            self.waiting_passengers.append(passenger)

            node = self.find_closest_node(passenger[1], passenger[2])
            if node not in self.passenger_nodes.keys():
                self.passenger_nodes[node] = deque()
            self.passenger_nodes[node].append(passenger)

        if (self.timestamp == passenger_re_entry_time):
            passenger = heapq.heappop(self.passenger_re_entry)
            self.waiting_passengers.append(passenger)

            node = self.find_closest_node(passenger[1], passenger[2])
            if node not in self.passenger_nodes:
                self.passenger_nodes[node] = deque()
            self.passenger_nodes[node].append(passenger)

        if (self.timestamp == driver_time):
            driver = self.driver_queue.popleft()
            driver.append(driver[0])
            driver.append(driver[0])
            self.waiting_drivers.append(driver)

            node = self.find_closest_node(driver[1], driver[2])
            if node not in self.driver_nodes.keys():
                self.driver_nodes[node] = deque()
            self.driver_nodes[node].append(driver)
           
        if (self.timestamp == re_entry_time):
            driver = heapq.heappop(self.driver_re_entry)
            self.waiting_drivers.append(driver)

            node = self.find_closest_node(driver[1], driver[2])
            if node not in self.driver_nodes:
                self.driver_nodes[node] = deque()
            self.driver_nodes[node].append(driver)

        return self.try_match()

    
    def try_match(self):
        cutoff = 10
        d1_total = 0
        d2_total = 0

        # go through drivers
        while self.waiting_drivers and (self.waiting_passengers or self.sleeping_passengers > 0):
            driver = self.waiting_drivers.popleft()
            node = self.find_closest_node(driver[1], driver[2])

            passenger_node, dist = self.find_closest_passenger_node(node, cutoff / 60)

            if passenger_node != None:
                # passenger found
                passenger = self.passenger_nodes[passenger_node].popleft()
                if len(self.passenger_nodes[passenger_node]) == 0: self.passenger_nodes.pop(passenger_node)

                if passenger in self.waiting_passengers: self.waiting_passengers.remove(passenger)
                else: self.sleeping_passengers -= 1

                self.driver_nodes[node].remove(driver)
                if len(self.driver_nodes[node]) == 0: self.driver_nodes.pop(node)

                d1, d2 = self.assign_ride(driver, passenger, dist * 60)
                d1_total += d1
                d2_total += d2
            else:
                # no driver found
                self.sleeping_drivers += 1

        # go through passengers
        while self.waiting_passengers and (self.waiting_drivers or self.sleeping_drivers > 0):
            passenger = self.waiting_passengers.popleft()
            node = self.find_closest_node(passenger[1], passenger[2])

            driver_node, dist = self.find_closest_driver_node(node, cutoff / 60)

            if driver_node != None:
                # driver found
                driver = self.driver_nodes[driver_node].popleft()
                if len(self.driver_nodes[driver_node]) == 0: self.driver_nodes.pop(driver_node)

                if driver in self.waiting_drivers: self.waiting_drivers.remove(driver)
                else: self.sleeping_drivers -= 1

                self.passenger_nodes[node].remove(passenger)
                if len(self.passenger_nodes[node]) == 0: self.passenger_nodes.pop(node)

                d1, d2 = self.assign_ride(driver, passenger, dist * 60)
                d1_total += d1
                d2_total += d2
            else:
                # no driver found
                self.sleeping_passengers += 1
        
        return d1_total, d2_total

    def assign_ride(self, driver, passenger, min_to_pickup):

        passenger_source_node = self.find_closest_node(passenger[1], passenger[2])
        passenger_dest_node = self.find_closest_node(passenger[3], passenger[4])


        # float representing hours: pick up to drop off time
        delivery_time= 60 * self.calc_travel_time(passenger_source_node, passenger_dest_node, self.timestamp)

        # float representing hours
        trip_time = min_to_pickup + delivery_time

        # date time object
        arrival_time = self.timestamp + timedelta(minutes=trip_time)

        # drivers log off starting after 4 hours of being logged on
        # drivers always log off after finsihing a ride if they've been logged on for more than 9 hours
        hours_logged = (arrival_time - driver[3]).total_seconds() / 3600
        if random.random() > ((hours_logged - 4) / 5):
            # set the new entry time, lat, and lon for driver as the drop off of the previous passenger
            driver[0] = arrival_time
            driver[4] = arrival_time
            driver[1] = self.nodes[passenger_dest_node]['lat']
            driver[2] = self.nodes[passenger_dest_node]['lon']
            heapq.heappush(self.driver_re_entry, driver)

        # for benchmarking
        passenger_wait = (self.timestamp - passenger[5]).total_seconds() / 60
        d1 = passenger_wait + trip_time
        d2 = delivery_time - min_to_pickup

        self.total_delivery_time += delivery_time
        self.total_pickup_time += min_to_pickup
        self.total_passenger_match_wait += passenger_wait

        return d1, d2

    def find_closest_node(self, lat, lon):
        return self.tree_nearest_node(lat, lon)
    
    def manhattan_dist(self, source_node, dest_node):
        source_coords = self.nodes[source_node]
        dest_coords = self.nodes[dest_node]
        units = abs(source_coords['lat'] - dest_coords['lat'])
        units += abs(source_coords['lon'] - dest_coords['lon'])
        return units * 65
    
    def a_star(self, source, dest, day_hour):
        def h(source_node):
            if (self.heuristic_mph == float('inf')): return 0
            miles = self.manhattan_dist(source_node, dest)
            return miles / self.heuristic_mph

        pq = [(h(source), 0, source)]  # Priority queue as a min-heap with (distance, node)
        visited = set()
        cost_so_far = {}
        cost_so_far[source] = 0

        while pq:
            (hscore, dist, current_node) = heapq.heappop(pq)

            if current_node == dest:
                return dist

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self.adjacency.get(current_node, {}):
                new_cost = dist + self.adjacency[current_node][neighbor][day_hour]
                if neighbor not in cost_so_far.keys() or new_cost < cost_so_far[neighbor]:
                    heapq.heappush(pq, (new_cost + h(neighbor), new_cost, neighbor))
                    cost_so_far[neighbor] = new_cost

        return float('inf')

    
    def find_closest_driver_node(self, passenger_node, cutoff):
        def datetime_to_string(dt):
            if dt.weekday() >= 5:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            hour = dt.hour
            return f"{day_type}_{hour}"
        
        day_hour = datetime_to_string(self.timestamp)
        pq = [(0, passenger_node)]  # Priority queue as a min-heap with (distance, node)
        visited = set()

        while pq:
            (dist, current_node) = heapq.heappop(pq)

            if current_node in self.driver_nodes.keys() and len(self.driver_nodes[current_node]) > 0:
                return current_node, dist

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self.adjacency.get(current_node, {}):
                if neighbor not in visited:
                    newcost = dist + self.adjacency[neighbor][current_node][day_hour]  
                    if newcost < cutoff: heapq.heappush(pq, (newcost, neighbor))
        return None, None
    
    def find_closest_passenger_node(self, driver_node, cutoff):
        def datetime_to_string(dt):
            if dt.weekday() >= 5:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            hour = dt.hour
            return f"{day_type}_{hour}"
        
        day_hour = datetime_to_string(self.timestamp)
        pq = [(0, driver_node)]  # Priority queue as a min-heap with (distance, node)
        visited = set()

        while pq:
            (dist, current_node) = heapq.heappop(pq)

            if current_node in self.passenger_nodes.keys():
                while len(self.passenger_nodes[current_node]) > 0:
                    passenger = self.passenger_nodes[current_node].popleft()
                    passenger_wait = (self.timestamp - passenger[5]).total_seconds() / 60
                    if passenger_wait < 20:
                        self.passenger_nodes[current_node].appendleft(passenger)
                        return current_node, dist
                    else:
                        if passenger in self.waiting_passengers :
                            self.waiting_passengers.remove(passenger)
                        else:
                            self.sleeping_passengers -= 1
                        self.passengers_left += 1

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self.adjacency.get(current_node, {}):
                if neighbor not in visited:
                    newcost = dist + self.adjacency[current_node][neighbor][day_hour]  
                    if newcost < cutoff: heapq.heappush(pq, (newcost, neighbor))

        return None, None
    


    def calc_travel_time(self, source_node, dest_node, current_time):
        def datetime_to_string(dt):
            if dt.weekday() >= 5:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            hour = dt.hour
            return f"{day_type}_{hour}"
        
        return self.a_star(source_node, dest_node, datetime_to_string(current_time))



# experiment

start_time = datetime.now()
print("starting at ", start_time)

not_uber = NotUber()
not_uber.heuristic_mph = 30 # set the miles per hour to use in the heuristic function
print("setting heuristic mph to", not_uber.heuristic_mph)

preprocess_time = datetime.now()
print("preprocessing done in " + str((preprocess_time - start_time).total_seconds() / 60))


d1_total = 0
d2_total = 0

num_passengers = len(not_uber.passenger_queue)
num_drivers = len(not_uber.driver_queue)

tick = 0
while not_uber.passenger_queue or not_uber.driver_queue or not_uber.driver_re_entry:
    d1, d2 = not_uber.update_timestamp()
    if d1 : d1_total += d1
    if d2 : d2_total += d2
    tick += 1
    if tick % 2000 == 0:
        print()
        print(not_uber.timestamp)
        print("waiting drivers:\t", len(not_uber.waiting_drivers))
        print("re-entering drivers:\t", len(not_uber.driver_re_entry))
        print("sleeping drivers:\t", not_uber.sleeping_drivers)
        print("waiting passengers:\t", len(not_uber.waiting_passengers))
        print("sleeping passengers:\t", not_uber.sleeping_passengers)
        print("passengers left:\t", not_uber.passengers_left)
    if datetime(2014, 4, 27, 6) < not_uber.timestamp:
        print("sim over")
        break


end_time = datetime.now()
# duration in minutes
duration = (end_time - start_time).total_seconds() / 60
sim_duration = (end_time - preprocess_time).total_seconds() / 60

unmatched_passengers = len(not_uber.waiting_passengers) + not_uber.passengers_left + not_uber.sleeping_passengers
unmatched_drivers = len(not_uber.waiting_drivers) + not_uber.sleeping_drivers
num_rides = num_passengers - unmatched_passengers

print()

print("unmatched passengers:\t", unmatched_passengers)
print("unmatched drivers:\t", unmatched_drivers)
print("passengers left:\t", not_uber.passengers_left)

print()

print("average d1:\t", d1_total / num_rides)
print("average d2:\t", d2_total / num_rides)
print("total d2:\t", d2_total)

print()

print("average wait for match:\t", not_uber.total_passenger_match_wait / num_rides)
print("average pickup time:\t", not_uber.total_pickup_time / num_rides)
print("average delivery time:\t", not_uber.total_delivery_time / num_rides)

print()

print("total time:\t", duration)
print("sim time:\t", sim_duration)

print()