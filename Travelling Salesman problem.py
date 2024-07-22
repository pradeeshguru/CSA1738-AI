from itertools import permutations

def calculate_total_distance(graph, path):
    return sum(graph[path[i]][path[i+1]] for i in range(len(path)-1)) + graph[path[-1]][path[0]]

def travelling_salesman(graph):
    cities = list(graph.keys())
    min_distance = float('inf')
    best_path = []

    for perm in permutations(cities):
        current_distance = calculate_total_distance(graph, perm)
        if current_distance < min_distance:
            min_distance = current_distance
            best_path = perm

    return best_path, min_distance

# Example usage
graph = {
    'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25},
    'C': {'A': 15, 'B': 35, 'C': 0, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0}
}

best_path, min_distance = travelling_salesman(graph)
print("Best path:", best_path)
print("Minimum distance:", min_distance)
