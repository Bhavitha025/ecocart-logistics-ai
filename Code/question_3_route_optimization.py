# Question 3: Search Algorithms for Route Optimization

import os
import time
import math
import heapq
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


GRAPH_FOLDER = "../Graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)


def create_delivery_graph():
    graph = {
        "Warehouse": {"A": 4, "B": 2},
        "A": {"Warehouse": 4, "C": 5, "D": 10},
        "B": {"Warehouse": 2, "C": 3, "E": 8},
        "C": {"A": 5, "B": 3, "D": 4, "F": 6},
        "D": {"A": 10, "C": 4, "G": 11},
        "E": {"B": 8, "F": 2},
        "F": {"C": 6, "E": 2, "G": 3},
        "G": {"D": 11, "F": 3, "Customer": 5},
        "Customer": {"G": 5}
    }

    positions = {
        "Warehouse": (0, 0),
        "A": (2, 4),
        "B": (2, -1),
        "C": (5, 2),
        "D": (8, 5),
        "E": (6, -3),
        "F": (9, -1),
        "G": (12, 2),
        "Customer": (15, 2)
    }

    return graph, positions


def path_cost(graph, path):
    total = 0
    for i in range(len(path) - 1):
        total += graph[path[i]][path[i + 1]]
    return total


def heuristic(node, goal, positions):
    x1, y1 = positions[node]
    x2, y2 = positions[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dfs(graph, start, goal):
    start_time = time.perf_counter()
    stack = [(start, [start])]
    visited = set()
    nodes_explored = 0

    while stack:
        current, path = stack.pop()
        nodes_explored += 1

        if current == goal:
            return path, path_cost(graph, path), nodes_explored, time.perf_counter() - start_time

        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None, float("inf"), nodes_explored, time.perf_counter() - start_time


def bfs(graph, start, goal):
    start_time = time.perf_counter()
    queue = deque([(start, [start])])
    visited = set()
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        if current == goal:
            return path, path_cost(graph, path), nodes_explored, time.perf_counter() - start_time

        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None, float("inf"), nodes_explored, time.perf_counter() - start_time


def a_star(graph, positions, start, goal):
    start_time = time.perf_counter()
    queue = []
    heapq.heappush(queue, (0, start, [start], 0))
    visited = set()
    nodes_explored = 0

    while queue:
        estimated_cost, current, path, actual_cost = heapq.heappop(queue)
        nodes_explored += 1

        if current == goal:
            return path, actual_cost, nodes_explored, time.perf_counter() - start_time

        if current not in visited:
            visited.add(current)

            for neighbor, road_cost in graph[current].items():
                if neighbor not in visited:
                    new_actual_cost = actual_cost + road_cost
                    new_estimated_cost = new_actual_cost + heuristic(neighbor, goal, positions)
                    heapq.heappush(
                        queue,
                        (new_estimated_cost, neighbor, path + [neighbor], new_actual_cost)
                    )

    return None, float("inf"), nodes_explored, time.perf_counter() - start_time


def save_delivery_network(graph, positions, best_path):
    network = nx.Graph()

    for node, neighbors in graph.items():
        for neighbor, cost in neighbors.items():
            network.add_edge(node, neighbor, weight=cost)

    plt.figure(figsize=(10, 6))

    nx.draw(
        network,
        positions,
        with_labels=True,
        node_size=1800,
        font_size=9,
        font_weight="bold"
    )

    edge_labels = nx.get_edge_attributes(network, "weight")
    nx.draw_networkx_edge_labels(network, positions, edge_labels=edge_labels)

    route_edges = list(zip(best_path, best_path[1:]))
    nx.draw_networkx_edges(network, positions, edgelist=route_edges, width=4)

    plt.title("EcoCart Delivery Network and Best Route")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Question 3-plot1.png"), dpi=300)
    plt.close()


def save_path_cost_plot(results_df):
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Algorithm"], results_df["Path Cost"])
    plt.title("Path Cost Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Path Cost")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Question 3-plot2.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    graph, positions = create_delivery_graph()

    start = "Warehouse"
    goal = "Customer"

    algorithms = {
        "DFS": lambda: dfs(graph, start, goal),
        "BFS": lambda: bfs(graph, start, goal),
        "A*": lambda: a_star(graph, positions, start, goal)
    }

    results = []

    for name, function in algorithms.items():
        path, cost, nodes, runtime = function()

        results.append({
            "Algorithm": name,
            "Path": " -> ".join(path),
            "Path Cost": cost,
            "Nodes Explored": nodes,
            "Runtime": runtime
        })

    results_df = pd.DataFrame(results)

    print("\nQuestion 3 Results:")
    print(results_df)

    best_path = results_df.sort_values("Path Cost").iloc[0]["Path"].split(" -> ")

    save_delivery_network(graph, positions, best_path)
    save_path_cost_plot(results_df)

    print("\nGraphs saved in the Graphs folder.")