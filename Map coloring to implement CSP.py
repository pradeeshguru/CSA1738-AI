def is_valid(coloring, node, color, graph):
    for neighbor in graph[node]:
        if coloring.get(neighbor) == color:
            return False
    return True

def map_coloring(graph, colors, coloring={}, node_list=None):
    if node_list is None:
        node_list = list(graph.keys())
    
    if not node_list:
        return coloring

    node = node_list[0]
    remaining_nodes = node_list[1:]

    for color in colors:
        if is_valid(coloring, node, color, graph):
            coloring[node] = color
            result = map_coloring(graph, colors, coloring, remaining_nodes)
            if result:
                return result
            del coloring[node]
    
    return None

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

colors = ['Red', 'Green', 'Blue']
coloring = map_coloring(graph, colors)
print("Coloring:", coloring)
