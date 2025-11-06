# code_init.py
import numpy as np
import copy
class CodeInit:
    def __init__(self):
        pass

    def get_initial_code_1(self):
        """Return the initial code that will be evolved"""
        return """
import math

def load_data(filename):
    nodes = []
    demands = []
    capacity = 0
    distances = []
    depot = 0

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]  # 去除每行首尾空白

    # 解析维度和容量
    for line in lines:
        if line.startswith('DIMENSION'):
            num_nodes = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line.startswith('DEPOT_SECTION'):
            depot_line_idx = lines.index(line)
            depot = int(lines[depot_line_idx + 1].strip()) - 1

    # 解析坐标数据（更健壮的匹配方式）
    start_coords = None
    end_coords = None
    for i, line in enumerate(lines):
        if line == "NODE_COORD_SECTION":
            start_coords = i + 1
        elif line == "DEMAND_SECTION":
            end_coords = i
            break

    if start_coords is None or end_coords is None:
        raise ValueError("文件格式错误：缺少 NODE_COORD_SECTION 或 DEMAND_SECTION")

    for line in lines[start_coords:end_coords]:
        parts = line.split()
        node_id = int(parts[0]) - 1
        x, y = int(parts[1]), int(parts[2])
        nodes.append((x, y))

    # 解析需求数据
    start_demands = end_coords + 1
    end_demands = lines.index("DEPOT_SECTION")
    for line in lines[start_demands:end_demands]:
        parts = line.split()
        demand = int(parts[1])
        demands.append(demand)
    # 计算距离矩阵
    distances = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = euclidean_distance(nodes[i], nodes[j])
            distances[i][j] = dist
            distances[j][i] = dist

    return nodes, demands, capacity, distances, depot, num_nodes


def euclidean_distance(node1, node2):

    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def nearest_neighbor_heuristic(nodes, demands, capacity, distances, depot, num_nodes):

    unvisited = set(range(num_nodes))
    unvisited.remove(depot)
    routes = []

    while unvisited:
        current_route = [depot]
        load = 0
        current_node = depot
        while unvisited:
            # 找到最近的节点，且车辆负载不超过容量
            nearest_node = None
            nearest_distance = float('inf')
            for node in unvisited:
                if load + demands[node] <= capacity:
                    distance = distances[current_node][node]
                    if distance < nearest_distance:
                        nearest_node = node
                        nearest_distance = distance

            if nearest_node is None:
                break

            # 更新路径
            current_route.append(nearest_node)
            load += demands[nearest_node]
            unvisited.remove(nearest_node)
            current_node = nearest_node

        # 返回仓库
        current_route.append(depot)
        routes.append(current_route)

    return routes


def total_distance(routes, distances, depot):

    total_dist = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += distances[route[i]][route[i + 1]]
        # 返回仓库
        total_dist += distances[route[-1]][depot]
    return total_dist


def print_solution(routes, total_dist):

    print("Routes:")
    for route in routes:
        print(" -> ".join(map(str, [node + 1 for node in route])))
    print(f"Total distance: {total_dist}")


# 主函数
if __name__ == "__main__":
    # 加载数据
    nodes, demands, capacity, distances, depot, num_nodes = load_data("att48.vrp")

    # 求解CVRP
    routes = nearest_neighbor_heuristic(nodes, demands, capacity, distances, depot, num_nodes)

    # 计算并打印结果
    total_dist = total_distance(routes, distances, depot)
    print_solution(routes, total_dist)

    """

    def get_initial_code_2(self):
        """Return the initial code that will be evolved"""
        return """
import math

def load_data(filename):
    nodes = []
    demands = []
    capacity = 0
    distances = []
    depot = 0

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]  # 去除每行首尾空白

    # 解析维度和容量
    for line in lines:
        if line.startswith('DIMENSION'):
            num_nodes = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line.startswith('DEPOT_SECTION'):
            depot_line_idx = lines.index(line)
            depot = int(lines[depot_line_idx + 1].strip()) - 1

    # 解析坐标数据（更健壮的匹配方式）
    start_coords = None
    end_coords = None
    for i, line in enumerate(lines):
        if line == "NODE_COORD_SECTION":
            start_coords = i + 1
        elif line == "DEMAND_SECTION":
            end_coords = i
            break

    if start_coords is None or end_coords is None:
        raise ValueError("文件格式错误：缺少 NODE_COORD_SECTION 或 DEMAND_SECTION")

    for line in lines[start_coords:end_coords]:
        parts = line.split()
        node_id = int(parts[0]) - 1
        x, y = int(parts[1]), int(parts[2])
        nodes.append((x, y))

    # 解析需求数据
    start_demands = end_coords + 1
    end_demands = lines.index("DEPOT_SECTION")
    for line in lines[start_demands:end_demands]:
        parts = line.split()
        demand = int(parts[1])
        demands.append(demand)
    # 计算距离矩阵
    distances = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = euclidean_distance(nodes[i], nodes[j])
            distances[i][j] = dist
            distances[j][i] = dist

    return nodes, demands, capacity, distances, depot, num_nodes


def euclidean_distance(node1, node2):

    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def nearest_neighbor_heuristic(nodes, demands, capacity, distances, depot, num_nodes):

    unvisited = set(range(num_nodes))
    unvisited.remove(depot)
    routes = []

    while unvisited:
        current_route = [depot]
        load = 0
        current_node = depot
        while unvisited:
            # 找到最近的节点，且车辆负载不超过容量
            nearest_node = None
            nearest_distance = float('inf')
            for node in unvisited:
                if load + demands[node] <= capacity:
                    distance = distances[current_node][node]
                    if distance < nearest_distance:
                        nearest_node = node
                        nearest_distance = distance

            if nearest_node is None:
                break

            # 更新路径
            current_route.append(nearest_node)
            load += demands[nearest_node]
            unvisited.remove(nearest_node)
            current_node = nearest_node

        # 返回仓库
        current_route.append(depot)
        routes.append(current_route)

    return routes


def total_distance(routes, distances, depot):

    total_dist = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += distances[route[i]][route[i + 1]]
        # 返回仓库
        total_dist += distances[route[-1]][depot]
    return total_dist


def print_solution(routes, total_dist):

    print("Routes:")
    for route in routes:
        print(" -> ".join(map(str, [node + 1 for node in route])))
    print(f"Total distance: {total_dist}")


# 主函数
if __name__ == "__main__":
    # 加载数据
    nodes, demands, capacity, distances, depot, num_nodes = load_data("att48.vrp")

    # 求解CVRP
    routes = nearest_neighbor_heuristic(nodes, demands, capacity, distances, depot, num_nodes)

    # 计算并打印结果
    total_dist = total_distance(routes, distances, depot)
    print_solution(routes, total_dist)

    """