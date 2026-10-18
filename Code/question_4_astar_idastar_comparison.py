# Question 4: A* and IDA* Comparison

import os
import time
import math
import heapq

import matplotlib.pyplot as plt
import pandas as pd


GRAPH_FOLDER = "../Graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)


def distance(node1, node2, positions):
    x1, y1 = positions[node1]
    x2, y2 = positions[node2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_cost(graph, path):
    total = 0
    for i in range(len(path) - 1):
        total += graph[path[i]][path[i + 1]]
    return total


def create_urban_graph():
    graph = {
        "Warehouse": {"U1": 2, "U2": 3},
        "U1": {"Warehouse": 2, "U3": 2, "U4": 4},
        "U2": {"Warehouse": 3, "U4": 2, "U5": 5},
        "U3": {"U1": 2, "U6": 3},
        "U4": {"U1": 4, "U2": 2, "U6": 2, "U7": 4},
        "U5": {"U2": 5, "U7": 2},
        "U6": {"U3": 3, "U4": 2, "Customer": 5},
        "U7": {"U4": 4, "U5": 2, "Customer": 3},
        "Customer": {"U6": 5, "U7": 3}
    }

    positions = {
        "Warehouse": (0, 0),
        "U1": (2, 2),
        "U2": (2, -2),
        "U3": (5, 3),
        "U4": (5, 0),
        "U5": (5, -3),
        "U6": (8, 2),
        "U7": (8, -2),
        "Customer": (11, 0)
    }

    return graph, positions


def create_rural_graph():
    graph = {
        "Warehouse": {"R1": 6, "R2": 9},
        "R1": {"Warehouse": 6, "R3": 8},
        "R2": {"Warehouse": 9, "R3": 5, "R4": 12},
        "R3": {"R1": 8, "R2": 5, "R5": 7},
        "R4": {"R2": 12, "R5": 6},
        "R5": {"R3": 7, "R4": 6, "Customer": 10},
        "Customer": {"R5": 10}
    }

    positions = {
        "Warehouse": (0, 0),
        "R1": (3, 5),
        "R2": (4, -4),
        "R3": (8, 1),
        "R4": (9, -6),
        "R5": (13, -1),
        "Customer": (18, 0)
    }

    return graph, positions


def a_star(graph, positions, start, goal):
    start_time = time.perf_counter()
    queue = []
    heapq.heappush(queue, (0, start, [start], 0))
    visited = set()
    nodes_explored = 0

    while queue:
        estimated_total, current, path, actual_cost = heapq.heappop(queue)
        nodes_explored += 1

        if current == goal:
            return path, actual_cost, nodes_explored, time.perf_counter() - start_time

        if current not in visited:
            visited.add(current)

            for neighbor, cost in graph[current].items():
                if neighbor not in visited:
                    new_actual_cost = actual_cost + cost
                    new_estimated_total = new_actual_cost + distance(neighbor, goal, positions)
                    heapq.heappush(
                        queue,
                        (new_estimated_total, neighbor, path + [neighbor], new_actual_cost)
                    )

    return None, float("inf"), nodes_explored, time.perf_counter() - start_time


def ida_star(graph, positions, start, goal):
    start_time = time.perf_counter()
    nodes_explored = 0

    def search(path, actual_cost, threshold):
        nonlocal nodes_explored

        current = path[-1]
        nodes_explored += 1

        estimated_total = actual_cost + distance(current, goal, positions)

        if estimated_total > threshold:
            return estimated_total, None

        if current == goal:
            return True, path

        minimum = float("inf")

        for neighbor, cost in graph[current].items():
            if neighbor not in path:
                result, found_path = search(
                    path + [neighbor],
                    actual_cost + cost,
                    threshold
                )

                if result is True:
                    return True, found_path

                minimum = min(minimum, result)

        return minimum, None

    threshold = distance(start, goal, positions)

    while True:
        result, path = search([start], 0, threshold)

        if result is True:
            return path, calculate_cost(graph, path), nodes_explored, time.perf_counter() - start_time

        if result == float("inf"):
            return None, float("inf"), nodes_explored, time.perf_counter() - start_time

        threshold = result


def save_comparison_plots(results_df):
    path_cost_table = results_df.pivot(
        index="Scenario",
        columns="Algorithm",
        values="Path Cost"
    )

    path_cost_table.plot(kind="bar", figsize=(8, 5))
    plt.title("A* vs IDA*: Path Cost")
    plt.xlabel("Route Scenario")
    plt.ylabel("Path Cost")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Question 4-plot1.png"), dpi=300)
    plt.close()

    nodes_table = results_df.pivot(
        index="Scenario",
        columns="Algorithm",
        values="Nodes Explored"
    )

    nodes_table.plot(kind="bar", figsize=(8, 5))
    plt.title("A* vs IDA*: Nodes Explored")
    plt.xlabel("Route Scenario")
    plt.ylabel("Nodes Explored")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Question 4-plot2.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    scenarios = {
        "Urban": create_urban_graph(),
        "Rural": create_rural_graph()
    }

    results = []

    for scenario_name, data in scenarios.items():
        graph, positions = data

        algorithms = {
            "A*": lambda: a_star(graph, positions, "Warehouse", "Customer"),
            "IDA*": lambda: ida_star(graph, positions, "Warehouse", "Customer")
        }

        for algorithm_name, algorithm in algorithms.items():
            path, cost, nodes, runtime = algorithm()

            results.append({
                "Scenario": scenario_name,
                "Algorithm": algorithm_name,
                "Path": " -> ".join(path),
                "Path Cost": cost,
                "Nodes Explored": nodes,
                "Runtime": runtime
            })

    results_df = pd.DataFrame(results)

    print("\nQuestion 4 Results:")
    print(results_df)

    save_comparison_plots(results_df)

    print("\nGraphs saved in the Graphs folder.")