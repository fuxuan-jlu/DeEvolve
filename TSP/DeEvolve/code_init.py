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
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix


def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)


def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    if len(unvisited_nodes) == 0:
        return destination_node

    # 计算直接到终点的距离
    direct_dist_to_dest = distance_matrix[current_node, destination_node]

    detour_factors = []
    connectivity_weights = []

    for node in unvisited_nodes:
        # 计算绕行因子
        detour = (distance_matrix[current_node, node] +
                  distance_matrix[node, destination_node] -
                  direct_dist_to_dest)
        detour_factors.append(detour)

        # 计算连接性权重
        remaining_nodes = np.setdiff1d(unvisited_nodes, node)
        if len(remaining_nodes) > 0:
            connectivity = np.sum(distance_matrix[node, remaining_nodes])
            connectivity_weights.append(connectivity)
        else:
            connectivity_weights.append(0)

    # 转换为numpy数组
    detour_factors = np.array(detour_factors)
    connectivity_weights = np.array(connectivity_weights)

    # 归一化处理
    normalized_detour = 1 / (1 + detour_factors)

    if np.max(connectivity_weights) > 0:
        normalized_connectivity = connectivity_weights / np.max(connectivity_weights)
    else:
        normalized_connectivity = np.ones_like(connectivity_weights)

    # 计算综合得分
    scores = 0.6 * normalized_detour + 0.4 * normalized_connectivity

    # 选择得分最高的节点
    next_node = unvisited_nodes[np.argmax(scores)]

    return next_node


def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length


if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "ali535.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""

    def get_initial_code_2(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)


def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    # 如果没有未访问节点，直接返回目标节点
    if len(unvisited_nodes) == 0:
        return destination_node

    # 获取当前节点到所有其他节点的距离向量
    current_pos = distance_matrix[current_node]
    # 获取目标节点到所有其他节点的距离向量
    dest_pos = distance_matrix[destination_node]

    scores = []  # 存储每个节点的评分

    for node in unvisited_nodes:
        # 计算当前节点到候选节点的距离
        dist = current_pos[node]
        # 计算候选节点到目标节点的方向性指标（欧氏距离）
        direction = np.linalg.norm(dest_pos - distance_matrix[node])
        # 综合评分：距离权重70%，方向权重30%
        score = 0.7 * (1 / dist) + 0.3 * (1 / direction)
        scores.append(score)

    # 选择评分最高的节点
    next_node = unvisited_nodes[np.argmax(scores)]
    return next_node
def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""
    def get_initial_code_3(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)

def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # 如果没有未访问节点，直接返回目标节点
    if len(unvisited_nodes) == 0:
        return destination_node

    # 计算当前节点到目标节点的直接距离
    direct_dist = distance_matrix[current_node, destination_node]

    detour_factors = []  # 存储每个节点的综合绕行因子

    for node in unvisited_nodes:
        # 计算绕行该节点的额外距离
        detour = (distance_matrix[current_node, node] +
                  distance_matrix[node, destination_node] -
                  direct_dist)

        # 计算邻居权重（该节点与未访问节点的连接紧密程度）
        neighbor_threshold = np.mean(distance_matrix[node])
        close_neighbors = np.sum(distance_matrix[node, unvisited_nodes] < neighbor_threshold)
        neighbor_weight = close_neighbors / len(unvisited_nodes)

        # 综合绕行因子：绕行距离 × (1 - 邻居权重)
        detour_factors.append(detour * (1 - neighbor_weight))

    # 选择综合绕行因子最小的节点
    min_idx = np.argmin(detour_factors)
    next_node = unvisited_nodes[min_idx]

    return next_node
def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""
    def get_initial_code_4(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)

def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    # 如果没有未访问节点，直接返回目标节点
    if len(unvisited_nodes) == 0:
        return destination_node

    # 初始化角度列表
    angles = []

    # 计算每个未访问节点的角度
    for node in unvisited_nodes:
        # 计算三角形三边距离
        dx1 = distance_matrix[current_node, node]  # 当前节点到候选节点距离
        dy1 = distance_matrix[node, destination_node]  # 候选节点到目标节点距离
        dx2 = distance_matrix[current_node, destination_node]  # 当前节点到目标节点直接距离

        # 使用余弦定理计算角度（添加小量1e-10防止除以零）
        numerator = dx1 ** 2 + dx2 ** 2 - dy1 ** 2
        denominator = 2 * dx1 * dx2 + 1e-10
        angle = np.arccos(numerator / denominator)
        angles.append(angle)

    # 转换为numpy数组
    angles = np.array(angles)

    # 计算接近度分数（当前节点到各未访问节点的距离）
    proximity = distance_matrix[current_node, unvisited_nodes]

    # 计算方向性分数（归一化角度）
    if np.max(angles) > 0:
        directionality = angles / np.max(angles)
    else:
        directionality = angles

    # 综合评分（60%距离权重 + 40%方向权重）
    combined_score = 0.6 * proximity + 0.4 * directionality

    # 选择综合评分最小的节点
    next_node = unvisited_nodes[np.argmin(combined_score)]

    return next_node

def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""
    def get_initial_code_5(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)

def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    # 如果没有未访问节点，直接返回目标节点
    if len(unvisited_nodes) == 0:
        return destination_node

    # 获取当前节点到所有未访问节点的距离向量
    current_to_unvisited = distance_matrix[current_node, unvisited_nodes]

    # 获取所有未访问节点到目标节点的距离向量
    unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # 设置权重参数（可调整）
    proximity_weight = 0.7  # 距离权重
    directionality_weight = 0.3  # 方向性权重

    # 计算综合评分：
    # 1. 距离因素：当前到候选节点的距离
    # 2. 方向因素：(当前到候选)-(候选到目标)，值越小表示方向越正确
    scores = (proximity_weight * current_to_unvisited +
              directionality_weight * (current_to_unvisited - unvisited_to_dest))

    # 选择评分最小的节点
    next_node_idx = np.argmin(scores)
    next_node = unvisited_nodes[next_node_idx]

    return next_node
def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""
    def get_initial_code_6(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)

def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    # 如果没有未访问节点，直接返回目标节点
    if len(unvisited_nodes) == 0:
        return destination_node

    # 获取当前节点到所有未访问节点的距离
    distances = distance_matrix[current_node, unvisited_nodes]
    # 获取目标节点到所有未访问节点的距离
    dest_distances = distance_matrix[destination_node, unvisited_nodes]

    # 归一化处理（添加1e-10防止除以零）
    normalized_dist = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-10)
    normalized_dest_dist = (dest_distances - np.min(dest_distances)) / (
                np.max(dest_distances) - np.min(dest_distances) + 1e-10)

    # 计算综合权重（60%当前距离 + 40%终点距离）
    # 使用1-归一化值，使得距离越小权重越大
    weights = 0.6 * (1 - normalized_dist) + 0.4 * (1 - normalized_dest_dist)

    # 选择权重最大的节点
    next_node = unvisited_nodes[np.argmax(weights)]

    return next_node
def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""
    def get_initial_code_7(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)

def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    # If no unvisited nodes left, return destination
    if len(unvisited_nodes) == 0:
        return destination_node

    # Remove current node from unvisited nodes if present
    current_pos = np.where(unvisited_nodes == current_node)[0]
    if len(current_pos) > 0:
        unvisited_nodes = np.delete(unvisited_nodes, current_pos)

    # Get distances from current node and to destination
    distances = distance_matrix[current_node, unvisited_nodes]
    dest_distances = distance_matrix[destination_node, unvisited_nodes]

    # Calculate progress scores (how much closer to destination)
    progress_scores = distance_matrix[current_node, destination_node] - dest_distances

    # Normalize both distance and progress metrics
    normalized_distances = distances / np.max(distances) if np.max(distances) > 0 else distances
    normalized_progress = progress_scores / np.max(progress_scores) if np.max(progress_scores) > 0 else progress_scores

    # Combine scores with weights (60% distance, 40% progress)
    combined_scores = 0.6 * normalized_distances + 0.4 * normalized_progress

    # Select node with minimum combined score
    next_node = unvisited_nodes[np.argmin(combined_scores)]

    return next_node
def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""
    def get_initial_code_8(self):
        """Return the initial code that will be evolved"""
        return """
import math
import numpy as np

# Global variables
N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    global N, COORDS, DIST
    reading_coords = False
    coords_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DIMENSION'):
                N = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line.startswith('EOF'):
                break
            elif reading_coords:
                parts = line.split()
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords_dict[node] = (x, y)

    # Convert to 0-based index
    COORDS = [coords_dict[i + 1] for i in range(N)]

    # Precompute distance matrix
    DIST = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = COORDS[i][0] - COORDS[j][0]
            dy = COORDS[i][1] - COORDS[j][1]
            DIST[i][j] = math.sqrt(dx * dx + dy * dy)

def tour_length(route):
    total = 0.0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += DIST[route[i]][route[j]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):

    # Handle case when no unvisited nodes remain
    if len(unvisited_nodes) == 0:
        return destination_node

    # Get distance vectors from current and destination nodes
    current_pos = distance_matrix[current_node]
    dest_pos = distance_matrix[destination_node]

    # Calculate direction vector to destination
    direction = dest_pos - current_pos

    # Get positions of all unvisited nodes
    unvisited_pos = distance_matrix[unvisited_nodes]

    # Calculate directional similarity (cosine similarity)
    displacement = unvisited_pos - current_pos
    dir_similarity = np.dot(displacement, direction) / (
            np.linalg.norm(displacement, axis=1) * np.linalg.norm(direction) + 1e-8)

    # Calculate proximity (inverse distance)
    proximity = 1 / (current_pos[unvisited_nodes] + 1e-8)

    # Combine scores with weights (60% proximity, 40% direction)
    combined_score = 0.6 * proximity + 0.4 * dir_similarity

    # Select node with highest combined score
    next_node = unvisited_nodes[np.argmax(combined_score)]

    return next_node
def solver():
    # Convert distance matrix to numpy array
    dist_matrix = np.array(DIST)

    # Initialize tour starting and ending at node 0
    start_node = 0
    current_node = start_node
    unvisited_nodes = set(range(N)) - {current_node}
    route = [current_node]

    # Build the tour using the select_next_node algorithm
    while unvisited_nodes:
        next_node = select_next_node(
            current_node,
            start_node,  # destination is the start node (to complete the loop)
            np.array(list(unvisited_nodes)),
            dist_matrix
        )
        route.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node

    # Complete the tour by returning to start
    route.append(start_node)

    length = tour_length(route)
    return route, length

if __name__ == "__main__":
    # Load the TSP data
    tsp_file = "a280.tsp"  # Change to your file path
    load_tsp_file(tsp_file)
    print(f"Loaded {N} cities")

    # Solve and print results
    route, length = solver()
    print("Best tour:", route)
    print("Tour length:", length)
"""