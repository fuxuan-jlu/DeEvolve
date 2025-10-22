"""
import numpy as np
import copy
import random
from itertools import combinations


class Individual:
    def __init__(self, truck_route, uav_assignments):
        self.truck_route = truck_route
        self.uav_assignments = uav_assignments
        self.fitness = float('inf')


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def initialize_population(points, population_size, depot=0):
    population = []
    nodes = [i for i in range(len(points)) if i != depot]

    for _ in range(population_size):
        truck_route = [depot] + random.sample(nodes, len(nodes)) + [depot]
        uav_assignments = {}

        # Randomly assign some nodes to UAV
        for node in nodes:
            if random.random() < 0.3:  # 30% chance of UAV assignment
                possible_launch = [i for i in truck_route if i != node]
                if possible_launch:
                    launch = random.choice(possible_launch)
                    uav_assignments[node] = launch

        population.append(Individual(truck_route, uav_assignments))

    return population


def calculate_fitness(individual, points, truck_speed, uav_speed, endurance, s_l, s_r):
    truck_times = {0: 0}
    uav_times = {}
    total_time = 0

    # Calculate truck travel times
    for i in range(1, len(individual.truck_route)):
        prev_node = individual.truck_route[i - 1]
        curr_node = individual.truck_route[i]
        dist = manhattan_distance(points[prev_node], points[curr_node])
        truck_times[curr_node] = truck_times[prev_node] + dist / truck_speed

    # Calculate UAV delivery times
    for node, launch in individual.uav_assignments.items():
        dist_to_node = euclidean_distance(points[launch], points[node])
        dist_to_next = euclidean_distance(points[node],
                                          points[individual.truck_route[individual.truck_route.index(launch) + 1]])

        if dist_to_node + dist_to_next > endurance:
            return float('inf')  # Invalid solution

        uav_time = truck_times[launch] + s_l + (dist_to_node + dist_to_next) / uav_speed + s_r
        uav_times[node] = uav_time

    # Calculate total completion time
    completion_time = truck_times[individual.truck_route[-1]]
    for node in individual.truck_route:
        if node in individual.uav_assignments.values():
            # Check if UAV returns in time
            assigned_nodes = [n for n, l in individual.uav_assignments.items() if l == node]
            for n in assigned_nodes:
                if uav_times[n] > truck_times[individual.truck_route[individual.truck_route.index(node) + 1]]:
                    return float('inf')  # UAV doesn't return in time

    return completion_time


def tournament_selection(population, tournament_size):
    selected = random.sample(population, tournament_size)
    return min(selected, key=lambda x: x.fitness)


def ordered_crossover(parent1, parent2):
    size = len(parent1.truck_route)
    start, end = sorted(random.sample(range(1, size - 1), 2))

    child1_route = [None] * size
    child2_route = [None] * size

    # Copy segments
    child1_route[start:end + 1] = parent1.truck_route[start:end + 1]
    child2_route[start:end + 1] = parent2.truck_route[start:end + 1]

    # Fill remaining positions
    pointer = 1
    for node in parent2.truck_route[1:-1]:
        if node not in child1_route[start:end + 1]:
            while child1_route[pointer] is not None:
                pointer += 1
            child1_route[pointer] = node
            pointer += 1

    pointer = 1
    for node in parent1.truck_route[1:-1]:
        if node not in child2_route[start:end + 1]:
            while child2_route[pointer] is not None:
                pointer += 1
            child2_route[pointer] = node
            pointer += 1

    # Ensure depot at start and end
    child1_route[0] = child1_route[-1] = 0
    child2_route[0] = child2_route[-1] = 0

    # Combine UAV assignments
    child1_uav = {}
    child2_uav = {}

    for node in parent1.uav_assignments:
        if node in child1_route:
            child1_uav[node] = parent1.uav_assignments[node]

    for node in parent2.uav_assignments:
        if node in child2_route:
            child2_uav[node] = parent2.uav_assignments[node]

    return Individual(child1_route, child1_uav), Individual(child2_route, child2_uav)


def mutate(individual, mutation_rate, points, endurance):
    nodes = individual.truck_route[1:-1]

    # Mutation for truck route
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(nodes) + 1), 2)
        individual.truck_route[i], individual.truck_route[j] = individual.truck_route[j], individual.truck_route[i]

    # Mutation for UAV assignments
    for node in nodes:
        if random.random() < mutation_rate:
            if node in individual.uav_assignments:
                del individual.uav_assignments[node]
            else:
                possible_launch = [n for n in individual.truck_route if n != node]
                if possible_launch:
                    launch = random.choice(possible_launch)
                    dist_to_node = euclidean_distance(points[launch], points[node])
                    next_node = individual.truck_route[individual.truck_route.index(launch) + 1]
                    dist_to_next = euclidean_distance(points[node], points[next_node])

                    if dist_to_node + dist_to_next <= endurance:
                        individual.uav_assignments[node] = launch

    return individual


def local_search(individual, points, truck_speed, uav_speed, endurance, s_l, s_r):
    improved = True
    while improved:
        improved = False
        current_fitness = individual.fitness

        # Try swapping truck nodes
        for i, j in combinations(range(1, len(individual.truck_route) - 1), 2):
            new_individual = copy.deepcopy(individual)
            new_individual.truck_route[i], new_individual.truck_route[j] = new_individual.truck_route[j], \
            new_individual.truck_route[i]
            new_individual.fitness = calculate_fitness(new_individual, points, truck_speed, uav_speed, endurance, s_l,
                                                       s_r)

            if new_individual.fitness < current_fitness:
                individual = new_individual
                improved = True
                break

        # Try moving nodes between truck and UAV
        for node in individual.truck_route[1:-1]:
            new_individual = copy.deepcopy(individual)

            if node in new_individual.uav_assignments:
                del new_individual.uav_assignments[node]
            else:
                possible_launch = [n for n in new_individual.truck_route if n != node]
                if possible_launch:
                    launch = random.choice(possible_launch)
                    dist_to_node = euclidean_distance(points[launch], points[node])
                    next_node = new_individual.truck_route[new_individual.truck_route.index(launch) + 1]
                    dist_to_next = euclidean_distance(points[node], points[next_node])

                    if dist_to_node + dist_to_next <= endurance:
                        new_individual.uav_assignments[node] = launch

            new_individual.fitness = calculate_fitness(new_individual, points, truck_speed, uav_speed, endurance, s_l,
                                                       s_r)

            if new_individual.fitness < current_fitness:
                individual = new_individual
                improved = True
                break

    return individual


def hybrid_genetic_algorithm(points, truck_speed=1.0, uav_speed=2.0, endurance=15.0, s_l=0.5, s_r=0.5,
                             population_size=50, generations=100, tournament_size=5, mutation_rate=0.1):
    # Initialize population
    population = initialize_population(points, population_size)

    # Evaluate initial population
    for individual in population:
        individual.fitness = calculate_fitness(individual, points, truck_speed, uav_speed, endurance, s_l, s_r)

    best_individual = min(population, key=lambda x: x.fitness)

    for generation in range(generations):
        new_population = []

        # Elitism: keep the best individual
        new_population.append(copy.deepcopy(best_individual))

        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            # Crossover
            child1, child2 = ordered_crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, mutation_rate, points, endurance)
            child2 = mutate(child2, mutation_rate, points, endurance)

            # Local search
            child1 = local_search(child1, points, truck_speed, uav_speed, endurance, s_l, s_r)
            child2 = local_search(child2, points, truck_speed, uav_speed, endurance, s_l, s_r)

            # Evaluate
            child1.fitness = calculate_fitness(child1, points, truck_speed, uav_speed, endurance, s_l, s_r)
            child2.fitness = calculate_fitness(child2, points, truck_speed, uav_speed, endurance, s_l, s_r)

            new_population.extend([child1, child2])

        population = new_population[:population_size]
        current_best = min(population, key=lambda x: x.fitness)

        if current_best.fitness < best_individual.fitness:
            best_individual = copy.deepcopy(current_best)

        # Early stopping if no improvement
        if generation > 10 and current_best.fitness >= best_individual.fitness:
            break

    return best_individual


def run_delivery_simulation():
    points = [(0, 0), (84, 36), (38, 86), (88, 65), (73, 57), (60, 88), (5, 63), (23, 59), (33, 71), (52, 17), (7, 40),
              (25, 89),
              (31, 83), (78, 43), (19, 49), (19, 26), (21, 6), (10, 33), (75, 35), (20, 81), (75, 65)]

    # Parameters
    truck_speed = 1.0
    uav_speed = 2.0
    endurance = 15.0
    s_l = 0.5
    s_r = 0.5

    # Run hybrid genetic algorithm
    best_solution = hybrid_genetic_algorithm(points, truck_speed, uav_speed, endurance, s_l, s_r)

    # Prepare output
    truck_route = best_solution.truck_route
    uav_assignments = best_solution.uav_assignments

    # Convert to subpath format similar to original
    subpaths = []
    current_subpath = [truck_route[0]]

    for i in range(1, len(truck_route)):
        node = truck_route[i]
        current_subpath.append(node)

        # Check if this node launches a UAV
        launching_nodes = [n for n, l in uav_assignments.items() if l == node]
        if launching_nodes:
            subpaths.append((current_subpath.copy(), -1))
            current_subpath = [node]

    if len(current_subpath) > 1:
        subpaths.append((current_subpath, -1))

    # Add UAV assignments to subpaths
    for subpath in subpaths:
        for node in subpath[0]:
            if node in uav_assignments:
                subpath = (subpath[0], node)

    # Calculate arrival times
    arrival_times = {0: 0}
    for i in range(1, len(truck_route)):
        prev_node = truck_route[i - 1]
        curr_node = truck_route[i]
        dist = manhattan_distance(points[prev_node], points[curr_node])
        arrival_times[curr_node] = arrival_times[prev_node] + dist / truck_speed

    # Adjust for UAV deliveries
    for node, launch in uav_assignments.items():
        dist_to_node = euclidean_distance(points[launch], points[node])
        next_node = truck_route[truck_route.index(launch) + 1]
        dist_to_next = euclidean_distance(points[node], points[next_node])

        uav_time = arrival_times[launch] + s_l + (dist_to_node + dist_to_next) / uav_speed + s_r
        arrival_times[node] = uav_time

    return points, subpaths, arrival_times


if __name__ == "__main__":
    points, trucksubpath, t = run_delivery_simulation()
    print("Delivery Points:", points)
    print("Truck Subpaths and UAV Usage:", trucksubpath)
    print("Arrival Times:", t)
"""
import numpy as np
import copy
import random
from itertools import combinations
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, truck_route, uav_assignments):
        self.truck_route = truck_route
        self.uav_assignments = uav_assignments
        self.fitness = float('inf')


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def initialize_population(points, population_size, depot=0):
    population = []
    nodes = [i for i in range(len(points)) if i != depot]

    for _ in range(population_size):
        truck_route = [depot] + random.sample(nodes, len(nodes)) + [depot]
        uav_assignments = {}

        # Randomly assign some nodes to UAV
        for node in nodes:
            if random.random() < 0.3:  # 30% chance of UAV assignment
                possible_launch = [i for i in truck_route if i != node]
                if possible_launch:
                    launch = random.choice(possible_launch)
                    uav_assignments[node] = launch

        population.append(Individual(truck_route, uav_assignments))

    return population


def calculate_fitness(individual, points, truck_speed, uav_speed, endurance, s_l, s_r):
    truck_times = {0: 0}
    uav_times = {}
    total_time = 0

    # Calculate truck travel times
    for i in range(1, len(individual.truck_route)):
        prev_node = individual.truck_route[i - 1]
        curr_node = individual.truck_route[i]
        dist = manhattan_distance(points[prev_node], points[curr_node])
        truck_times[curr_node] = truck_times[prev_node] + dist / truck_speed

    # Calculate UAV delivery times
    for node, launch in individual.uav_assignments.items():
        dist_to_node = euclidean_distance(points[launch], points[node])
        dist_to_next = euclidean_distance(points[node],
                                          points[individual.truck_route[individual.truck_route.index(launch) + 1]])

        if dist_to_node + dist_to_next > endurance:
            return float('inf')  # Invalid solution

        uav_time = truck_times[launch] + s_l + (dist_to_node + dist_to_next) / uav_speed + s_r
        uav_times[node] = uav_time

    # Calculate total completion time
    completion_time = truck_times[individual.truck_route[-1]]
    for node in individual.truck_route:
        if node in individual.uav_assignments.values():
            # Check if UAV returns in time
            assigned_nodes = [n for n, l in individual.uav_assignments.items() if l == node]
            for n in assigned_nodes:
                if uav_times[n] > truck_times[individual.truck_route[individual.truck_route.index(node) + 1]]:
                    return float('inf')  # UAV doesn't return in time

    return completion_time


def tournament_selection(population, tournament_size):
    selected = random.sample(population, tournament_size)
    return min(selected, key=lambda x: x.fitness)


def ordered_crossover(parent1, parent2):
    size = len(parent1.truck_route)
    start, end = sorted(random.sample(range(1, size - 1), 2))

    child1_route = [None] * size
    child2_route = [None] * size

    # Copy segments
    child1_route[start:end + 1] = parent1.truck_route[start:end + 1]
    child2_route[start:end + 1] = parent2.truck_route[start:end + 1]

    # Fill remaining positions
    pointer = 1
    for node in parent2.truck_route[1:-1]:
        if node not in child1_route[start:end + 1]:
            while child1_route[pointer] is not None:
                pointer += 1
            child1_route[pointer] = node
            pointer += 1

    pointer = 1
    for node in parent1.truck_route[1:-1]:
        if node not in child2_route[start:end + 1]:
            while child2_route[pointer] is not None:
                pointer += 1
            child2_route[pointer] = node
            pointer += 1

    # Ensure depot at start and end
    child1_route[0] = child1_route[-1] = 0
    child2_route[0] = child2_route[-1] = 0

    # Combine UAV assignments
    child1_uav = {}
    child2_uav = {}

    for node in parent1.uav_assignments:
        if node in child1_route:
            child1_uav[node] = parent1.uav_assignments[node]

    for node in parent2.uav_assignments:
        if node in child2_route:
            child2_uav[node] = parent2.uav_assignments[node]

    return Individual(child1_route, child1_uav), Individual(child2_route, child2_uav)


def mutate(individual, mutation_rate, points, endurance):
    nodes = individual.truck_route[1:-1]

    # Mutation for truck route
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(nodes) + 1), 2)
        individual.truck_route[i], individual.truck_route[j] = individual.truck_route[j], individual.truck_route[i]

    # Mutation for UAV assignments
    for node in nodes:
        if random.random() < mutation_rate:
            if node in individual.uav_assignments:
                del individual.uav_assignments[node]
            else:
                possible_launch = [n for n in individual.truck_route if n != node]
                if possible_launch:
                    launch = random.choice(possible_launch)
                    dist_to_node = euclidean_distance(points[launch], points[node])
                    next_node = individual.truck_route[individual.truck_route.index(launch) + 1]
                    dist_to_next = euclidean_distance(points[node], points[next_node])

                    if dist_to_node + dist_to_next <= endurance:
                        individual.uav_assignments[node] = launch

    return individual


def local_search(individual, points, truck_speed, uav_speed, endurance, s_l, s_r):
    improved = True
    while improved:
        improved = False
        current_fitness = individual.fitness

        # Try swapping truck nodes
        for i, j in combinations(range(1, len(individual.truck_route) - 1), 2):
            new_individual = copy.deepcopy(individual)
            new_individual.truck_route[i], new_individual.truck_route[j] = new_individual.truck_route[j], \
            new_individual.truck_route[i]
            new_individual.fitness = calculate_fitness(new_individual, points, truck_speed, uav_speed, endurance, s_l,
                                                       s_r)

            if new_individual.fitness < current_fitness:
                individual = new_individual
                improved = True
                break

        # Try moving nodes between truck and UAV
        for node in individual.truck_route[1:-1]:
            new_individual = copy.deepcopy(individual)

            if node in new_individual.uav_assignments:
                del new_individual.uav_assignments[node]
            else:
                possible_launch = [n for n in new_individual.truck_route if n != node]
                if possible_launch:
                    launch = random.choice(possible_launch)
                    dist_to_node = euclidean_distance(points[launch], points[node])
                    next_node = new_individual.truck_route[new_individual.truck_route.index(launch) + 1]
                    dist_to_next = euclidean_distance(points[node], points[next_node])

                    if dist_to_node + dist_to_next <= endurance:
                        new_individual.uav_assignments[node] = launch

            new_individual.fitness = calculate_fitness(new_individual, points, truck_speed, uav_speed, endurance, s_l,
                                                       s_r)

            if new_individual.fitness < current_fitness:
                individual = new_individual
                improved = True
                break

    return individual


def hybrid_genetic_algorithm(points, truck_speed=1.0, uav_speed=2.0, endurance=15.0, s_l=0.5, s_r=0.5,
                             population_size=50, generations=100, tournament_size=5, mutation_rate=0.1):
    # Initialize population
    population = initialize_population(points, population_size)

    # Evaluate initial population
    for individual in population:
        individual.fitness = calculate_fitness(individual, points, truck_speed, uav_speed, endurance, s_l, s_r)

    best_individual = min(population, key=lambda x: x.fitness)

    for generation in range(generations):
        new_population = []

        # Elitism: keep the best individual
        new_population.append(copy.deepcopy(best_individual))

        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            # Crossover
            child1, child2 = ordered_crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, mutation_rate, points, endurance)
            child2 = mutate(child2, mutation_rate, points, endurance)

            # Local search
            child1 = local_search(child1, points, truck_speed, uav_speed, endurance, s_l, s_r)
            child2 = local_search(child2, points, truck_speed, uav_speed, endurance, s_l, s_r)

            # Evaluate
            child1.fitness = calculate_fitness(child1, points, truck_speed, uav_speed, endurance, s_l, s_r)
            child2.fitness = calculate_fitness(child2, points, truck_speed, uav_speed, endurance, s_l, s_r)

            new_population.extend([child1, child2])

        population = new_population[:population_size]
        current_best = min(population, key=lambda x: x.fitness)

        if current_best.fitness < best_individual.fitness:
            best_individual = copy.deepcopy(current_best)

        # Early stopping if no improvement
        if generation > 10 and current_best.fitness >= best_individual.fitness:
            break

    return best_individual


def run_delivery_simulation():
    points = [(0, 0), (84, 36), (38, 86), (88, 65), (73, 57), (60, 88), (5, 63), (23, 59), (33, 71), (52, 17), (7, 40),
              (25, 89),
              (31, 83), (78, 43), (19, 49), (19, 26), (21, 6), (10, 33), (75, 35), (20, 81), (75, 65)]

    # Parameters
    truck_speed = 1.0
    uav_speed = 2.0
    endurance = 15.0
    s_l = 0.5
    s_r = 0.5

    # Run hybrid genetic algorithm
    best_solution = hybrid_genetic_algorithm(points, truck_speed, uav_speed, endurance, s_l, s_r)

    # Prepare output
    truck_route = best_solution.truck_route
    uav_assignments = best_solution.uav_assignments

    # Convert to subpath format similar to original
    subpaths = []
    current_subpath = [truck_route[0]]

    for i in range(1, len(truck_route)):
        node = truck_route[i]
        current_subpath.append(node)

        # Check if this node launches a UAV
        launching_nodes = [n for n, l in uav_assignments.items() if l == node]
        if launching_nodes:
            subpaths.append((current_subpath.copy(), -1))
            current_subpath = [node]

    if len(current_subpath) > 1:
        subpaths.append((current_subpath, -1))

    # Add UAV assignments to subpaths
    for subpath in subpaths:
        for node in subpath[0]:
            if node in uav_assignments:
                subpath = (subpath[0], node)

    # Calculate arrival times
    arrival_times = {0: 0}
    for i in range(1, len(truck_route)):
        prev_node = truck_route[i - 1]
        curr_node = truck_route[i]
        dist = manhattan_distance(points[prev_node], points[curr_node])
        arrival_times[curr_node] = arrival_times[prev_node] + dist / truck_speed

    # Adjust for UAV deliveries
    for node, launch in uav_assignments.items():
        dist_to_node = euclidean_distance(points[launch], points[node])
        next_node = truck_route[truck_route.index(launch) + 1]
        dist_to_next = euclidean_distance(points[node], points[next_node])

        uav_time = arrival_times[launch] + s_l + (dist_to_node + dist_to_next) / uav_speed + s_r
        arrival_times[node] = uav_time

    return points, subpaths, arrival_times


if __name__ == "__main__":
    points, trucksubpath, t = run_delivery_simulation()
    print("Delivery Points:", points)
    print("Truck Subpaths and UAV Usage:", trucksubpath)
    print("Arrival Times:", t)

    # Reorder trucksubpath for continuity
    ordered_subpaths = []
    remaining = list(trucksubpath)
    current_start = 0
    while remaining:
        found = False
        for sub in remaining:
            if sub[0][0] == current_start:
                ordered_subpaths.append(sub)
                current_start = sub[0][-1]
                remaining.remove(sub)
                found = True
                break
        if not found:
            break  # or handle error

    # If no continuous order was found or paths are unchanged, use original order
    if not ordered_subpaths:
        ordered_subpaths = trucksubpath

    # Path visualization
    plt.figure(figsize=(10, 8))
    for i, p in enumerate(points):
        color = 'green' if i == 0 else 'black'
        marker = 's' if i == 0 else 'o'
        plt.plot(p[0], p[1], marker=marker, color=color)
        plt.text(p[0] + 1, p[1] + 1, str(i), fontsize=12)

    # Add legend for warehouse
    plt.plot([], [], 'gs', label='Warehouse')

    # Construct full truck path by concatenating subpaths
    truck_points = []
    for sub in ordered_subpaths:
        for node in sub[0]:
            if not truck_points or truck_points[-1] != node:
                truck_points.append(node)

    # Plot truck path with arrows
    for idx in range(len(truck_points) - 1):
        start = truck_points[idx]
        end = truck_points[idx + 1]
        dx = points[end][0] - points[start][0]
        dy = points[end][1] - points[start][1]
        plt.arrow(points[start][0], points[start][1], dx, dy, head_width=1.5, width=0.1, color='blue', length_includes_head=True)

    # Add legend for truck path
    plt.plot([], [], 'b-', label='Truck Path')

    # Plot drone paths with arrows
    drone_label_added = False
    for sub in ordered_subpaths:
        if sub[1] != -1:
            launch = sub[0][0]
            rend = sub[0][-1]
            drone = sub[1]
            # Arrow from launch to drone
            dx1 = points[drone][0] - points[launch][0]
            dy1 = points[drone][1] - points[launch][1]
            plt.arrow(points[launch][0], points[launch][1], dx1, dy1, head_width=1.5, width=0.1, color='red', length_includes_head=True, linestyle='--')
            # Arrow from drone to rend
            dx2 = points[rend][0] - points[drone][0]
            dy2 = points[rend][1] - points[drone][1]
            plt.arrow(points[drone][0], points[drone][1], dx2, dy2, head_width=1.5, width=0.1, color='red', length_includes_head=True, linestyle='--')
            if not drone_label_added:
                plt.plot([], [], 'r--', label='Drone Path')
                drone_label_added = True

    plt.title('Truck-Drone Delivery Paths')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
