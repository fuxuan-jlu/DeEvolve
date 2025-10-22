import re
import sys
import time
import io
import math
from multiprocessing import Process, Queue

import numpy as np

points =[(0, 0), (55, 29), (75, 34), (44, 50), (46, 7), (17, 29), (79, 70), (75, 53),
         (81, 54), (90, 73), (83, 62), (48, 86), (37, 58), (19, 90), (27, 62), (9, 51), (70, 63), (84, 8), (63, 57), (87, 36), (68, 23)]


def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


# Function to generate Euclidean distance for UAV
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


truck_speed = 1.0  # Truck speed (units per time)
uav_speed = 2.0  # UAV speed (units per time)
e = 15.0  # UAV endurance
s_l = 0.5  # UAV launch time
s_r = 0.5  # UAV recovery time


def worker(code_str, queue):
    namespace = {'__name__': '__main__'}
    start_time = time.time()

    # 子进程内部重定向 stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(code_str, namespace, namespace)
        end_time = time.time()
        queue.put((
            namespace.get('trucksubpath'),
            None,
            end_time - start_time
        ))
    except Exception as e:
        queue.put((None, e, 0))
    finally:

        sys.stdout = old_stdout

def Evaluate(code_str):
    """
Extract and execute the Python code within the curly braces in `code_str`,
and return the float values of the runtime (seconds) and delivery completion time.
If the runtime exceeds 5 minutes, return a very large value and terminate the program.。
    """
    print("evaluate begin!")

    # 2. Clean up and prepare the code
    code_str = re.sub(r'^[\'"]|[\'"]$', '', code_str)
    code_str = code_str.encode().decode('unicode_escape')

    # 3. Prepare stdout redirection and get ready for timing
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    queue = Queue()
    p = Process(target=worker, args=(code_str, queue))
    p.start()

    # Non-blocking check to avoid a fixed 300-second wait
    start_time = time.time()
    timeout = 300
    while True:
        if not queue.empty():
            break
        if time.time() - start_time > timeout:
            p.terminate()
            #p.join()
            sys.stdout = old_stdout
            return (1500, 1500)
        time.sleep(0.1)  # 避免忙等待

    trucksubpath, error, run_time = queue.get()

    p.terminate()
    sys.stdout = old_stdout

    if error is not None or trucksubpath is None:
        return (1500, 1500)

    try:
        completion_time = get_com_time(trucksubpath, points, truck_speed, uav_speed, s_l, s_r)
    except Exception as e:
        raise ValueError(str(e))

    print("evaluate success!")
    return (run_time, completion_time)

def get_com_time(trucksubpath, delivery_points, truck_speed, uav_speed, s_l, s_r):
    truck_time = 0.0
    uav_time = 0.0
    visited_nodes = set()  # Used to record all visited nodes
    # Check if the truck path starts from point 0 (origin)
    if not trucksubpath or not trucksubpath[0][0] or trucksubpath[0][0][0] != 0:
        raise ValueError("Truck path must start from point 0 (origin)")

    for subpath in trucksubpath:
        nodes, uav_target = subpath
        truck_nodes = nodes  # Sequence of nodes visited by the truck

        # Record the nodes visited by the truck
        for node in truck_nodes:
            visited_nodes.add(node)

        # Truck travel time (time calculated in segments to reach each node)
        truck_arrival_times = [truck_time]  # Record the time each truck arrives at each node
        for i in range(len(truck_nodes) - 1):
            from_node = truck_nodes[i]
            to_node = truck_nodes[i + 1]
            distance = manhattan_distance(delivery_points[from_node], delivery_points[to_node])
            truck_arrival_times.append(truck_arrival_times[-1] + distance / truck_speed)

        truck_time = truck_arrival_times[-1]  # Time for the truck to complete the current subpath


        if uav_target != -1:
            visited_nodes.add(uav_target)  # Record the nodes visited by the drone
            launch_node = truck_nodes[0]  # The drone takes off from the first node of the sub-path.
            recovery_node = truck_nodes[-1]  # The drone returns to the last node of the sub-path

            # Drone launch time (the truck must have arrived at the launch node)
            uav_launch_time = truck_arrival_times[0]
            uav_time = max(uav_time, uav_launch_time) + s_l

            # The drone flies to the target node (using Euclidean distance)
            distance_to_target = euclidean_distance(delivery_points[launch_node], delivery_points[uav_target])
            uav_flight_time = distance_to_target / uav_speed
            uav_time += uav_flight_time

            # The drone returns from the target node to the recovery node (using Euclidean distance)
            distance_to_recovery = euclidean_distance(delivery_points[uav_target], delivery_points[recovery_node])
            uav_time += distance_to_recovery / uav_speed

            # Drone Recovery Time
            uav_time += s_r

            # The truck must wait for the drone to return (if the truck arrives first).
            truck_arrival_at_recovery = truck_arrival_times[-1]  # The time the truck arrives at the recycling node
            if uav_time > truck_arrival_at_recovery:
                truck_time = uav_time
            else:
                uav_time = truck_arrival_at_recovery

    # Check whether all nodes have been visited
    all_nodes = set(range(len(delivery_points)))
    unvisited_nodes = all_nodes - visited_nodes
    if unvisited_nodes:
        raise ValueError(f"The following nodes have not been visited: {sorted(unvisited_nodes)}")

    return max(truck_time, uav_time)



