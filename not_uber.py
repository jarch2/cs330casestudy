import csv
from queue import Queue
import heapq
from datetime import datetime, timedelta
import json

class NotUber:
    def __init__(self):
        def create_edges_dict(filename):
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
        self.passenger_queue = Queue()
        self.driver_queue = []
        self.nodes = self.load_json('node_data.json')
        self.adjacency = create_edges_dict('edges.csv')
        self.load_passengers('passengers.csv')
        self.load_drivers('drivers.csv')

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
        for passenger in passengers[1:]:
            self.passenger_queue.put(passenger)

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
        for driver in drivers[1:]:
            heapq.heappush(self.driver_queue, (driver[0], driver))

    def match_ride_t1(self):
        if not self.passenger_queue.empty() and self.driver_queue:
            passenger = self.passenger_queue.pop()
            _, driver = heapq.heappop(self.driver_queue)
            return self.assign_ride(driver, passenger)
        return None

    def assign_ride(self, driver, passenger):
        driver_node = self.find_closest_node(driver[1], driver[2])
        passenger_source_node = self.find_closest_node(passenger[1], passenger[2])
        passenger_dest_node = self.find_closest_node(passenger[3], passenger[4])
        travel_time_driver_to_passenger = self.calc_travel_time(driver_node, passenger_source_node, max(passenger[0], driver[0]))
        travel_time = self.calc_travel_time(passenger_source_node, passenger_dest_node, max(passenger[0], driver[0]))
        total_travel_time = travel_time_driver_to_passenger + travel_time
        arrival_time = max(passenger[0], driver[0]) + timedelta(hours=total_travel_time)

        # set the new entry time, lat, and lon for driver as the drop off of the previous passenger
        driver[0] = arrival_time
        driver[1] = self.nodes[passenger_dest_node]['lat']
        driver[2] = self.nodes[passenger_dest_node]['lon']
        heapq.heappush(self.driver_queue, (driver[0], driver))

        # for benchmarking
        d1 = (arrival_time - passenger[0]).total_seconds() / 60
        d2 = (travel_time - travel_time_driver_to_passenger).total_seconds() / 60
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


            


    
    def convert_string_to_datetime(self, date_string):
        date_format = "%m/%d/%Y %H:%M:%S"
        date_object = datetime.strptime(date_string, date_format)
        return date_object

    def load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file, encoding="utf-8")
        return data

    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return dlat**2 + dlon**2


not_uber = NotUber()
d1_total = 0
d2_total = 0
num_passengers = not_uber.passenger_queue.qsize()
while not not_uber.passenger_queue.empty():
    d1, d2 = not_uber.match_ride_t1()
    d1_total += d1
    d2_total += d2
print(d1_total/num_passengers, d2_total/num_passengers)