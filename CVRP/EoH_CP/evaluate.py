import re
import sys
import time
import io
import math
import traceback
from multiprocessing import Process, Queue

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
            dist = math.sqrt((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2)
            distances[i][j] = dist
            distances[j][i] = dist

    return nodes, demands, capacity, distances, depot, num_nodes


def euclidean_distance(node1, node2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

nodes, demands, capacity, distances, depot, num_nodes = load_data("att48.vrp")

def worker(code_str, queue):
    namespace = {'__name__': '__main__'}
    start_time = time.time()

    # 子进程内部重定向 stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()  # 重定向到内存缓冲区

    try:
        exec(code_str, namespace, namespace)
        print("yes!")
        end_time = time.time()
        queue.put((
            namespace.get('routes') or locals().get('routes'),
            None,
            end_time - start_time
        ))

    except Exception as e:
        sys.stdout = old_stdout
        queue.put((None, e, 0))
    finally:
        sys.stdout = old_stdout  # 恢复 stdout（可选，因为子进程会退出）

def Evaluate(code_str):
    """
    提取并执行 `code_str` 中花括号里的 Python 代码，
    返回  运行时间(秒) + 配送完成时间  的浮动值。

    如果运行时间超过5分钟，则返回极大值并结束程序。
    """
    print("evaluate begin!")

    # 2. 清理和准备代码
    code_str = re.sub(r'^[\'"]|[\'"]$', '', code_str)
    code_str = code_str.encode().decode('unicode_escape')
    # 3. 做好 stdout 重定向，准备计时
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    queue = Queue()
    p = Process(target=worker, args=(code_str, queue))
    p.start()

    # 非阻塞检查，避免固定等待 300 秒
    start_time = time.time()
    timeout = 300

    while True:

        if not queue.empty():
            break
        if time.time() - start_time > timeout:
            p.terminate()
            #p.join()
            sys.stdout = old_stdout
            return (60000, 60000)
        time.sleep(0.1)  # 避免忙等待
    routes, error, run_time = queue.get()

    #p.join()  # 确保子进程退出
    p.terminate()
    sys.stdout = old_stdout

    if error is not None or routes is None:

        return (60000, 60000)
    try:
        completion_time = get_com_time(routes, distances, demands, capacity, depot)
    except Exception as e:
        raise ValueError(str(e))
        #return 1000
    print("evaluate success!")
    return (run_time, completion_time)

def get_com_time(routes, distances, demands, capacity, depot):
    """
    计算总配送行程并检查是否满足CVRP的约束条件

    参数:
        routes (list): 车辆路径列表，每个路径是一个节点列表
        distances (list): 距离矩阵
        demands (list): 各节点的需求量
        capacity (int): 车辆容量
        depot (int): 仓库节点索引

    返回:
        float: 总配送行程

    异常:
        ValueError: 如果违反任何约束条件
    """
    total_dist = 0.0
    visited_nodes = set()

    # 检查每个路径是否以仓库开始和结束
    for route in routes:
        if route[0] != depot or route[-1] != depot:
            raise ValueError(f"路径 {route} 没有以仓库开始和结束")

    # 检查所有节点是否被访问且只被访问一次(除了仓库)
    for route in routes:
        for node in route[1:-1]:  # 跳过路径的首尾仓库节点
            if node == depot:
                raise ValueError(f"路径 {route} 中间包含了仓库节点")
            if node in visited_nodes:
                raise ValueError(f"节点 {node} 被多次访问")
            visited_nodes.add(node)

    # 检查是否所有非仓库节点都被访问
    all_nodes = set(range(len(demands)))
    required_nodes = all_nodes - {depot}
    if visited_nodes != required_nodes:
        missing_nodes = required_nodes - visited_nodes
        extra_nodes = visited_nodes - required_nodes
        error_msg = []
        if missing_nodes:
            error_msg.append(f"未访问的节点: {missing_nodes}")
        if extra_nodes:
            error_msg.append(f"多余访问的节点: {extra_nodes}")
        raise ValueError("; ".join(error_msg))

    # 检查车辆容量约束
    for i, route in enumerate(routes):
        load = 0
        for node in route[1:-1]:  # 只计算路径中的客户节点
            load += demands[node]
        if load > capacity:
            raise ValueError(f"路径 {i + 1} 的负载 {load} 超过了车辆容量 {capacity}")

    # 计算总距离
    total_dist = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += distances[route[i]][route[i + 1]]

    return total_dist

