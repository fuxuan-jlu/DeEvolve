import math
import os
import pickle
import re
import sys
import time
import io
from multiprocessing import Process, Queue

N = 0  # Number of cities
COORDS = []  # List of (x, y) coordinates
DIST = []  # Distance matrix

def load_tsp_file(filename):
    """Load TSP data from file in TSPLIB format"""
    global N, COORDS, DIST # 显式声明使用全局变量
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

tsp_file = "ali535.tsp"  # Change to your file path
load_tsp_file(tsp_file)
def worker(code_str, queue):
    namespace = {'__name__': '__main__'}
    start_time = time.time()

    # 子进程内部重定向 stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()  # 重定向到内存缓冲区

    try:
        exec(code_str, namespace, namespace)
        end_time = time.time()
        queue.put((
            namespace.get('route') or locals().get('route'),
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
            return (1000000, 1000000)
        time.sleep(0.1)  # 避免忙等待
    route, error, run_time = queue.get()

    #p.join()  # 确保子进程退出
    p.terminate()
    sys.stdout = old_stdout

    if error is not None or route is None:
        return (1000000, 1000000)
    try:
        completion_time = get_com_time(route, COORDS)
    except Exception as e:
        raise ValueError(str(e))
        #return 1000
    print("evaluate success!")
    return (run_time, completion_time)


def get_com_time(route , points):
    global  DIST  # 显式声明使用全局变量
    # 检查是否从起点出发
    if route[0] != 0:
        raise ValueError("未从起点出发")

    # 检查节点重复访问（起点0可以出现两次，其他节点只能出现一次）
    node_counts = {}
    for node in route:
        node_counts[node] = node_counts.get(node, 0) + 1
#node_counts.get(node, 0)：获取该节点当前的计数（若不存在则返回0）
    duplicate_nodes = []
    for node, count in node_counts.items():
        if node == 0:
            if count > 2:
                duplicate_nodes.append(node)
        else:
            if count > 1:
                duplicate_nodes.append(node)

    if duplicate_nodes:
        raise ValueError(f"以下节点被重复访问: {duplicate_nodes}")

    # 检查是否所有点都被访问（假设起点 0 是仓库，其余点必须被访问）
    visited = set(route)
    all_points = set(range(len(points)))
    missing_points = all_points - visited

    if missing_points:
        raise ValueError(f"以下配送点未被访问: {missing_points}")

    # 计算总行驶距离（包含返回起点的闭环）
    total_distance = 0.0
    N = len(route)

    for i in range(N):
        a = route[i]
        b = route[(i + 1) % N]  # 闭环处理
        total_distance += DIST[a][b]

    return total_distance

