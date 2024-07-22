from collections import deque

def is_valid_state(m, c):
    if m < 0 or c < 0 or m > 3 or c > 3 or (m != 0 and m < c) or (m != 3 and (3 - m) < (3 - c)):
        return False
    return True

def get_possible_moves(m, c, b):
    moves = []
    if b == 1:
        for i in range(1, min(m, 2) + 1):
            for j in range(0, min(c, i) + 1):
                if is_valid_state(m - i, c - j):
                    moves.append((m - i, c - j, 0))
    else:
        for i in range(1, min(3 - m, 2) + 1):
            for j in range(0, min(3 - c, i) + 1):
                if is_valid_state(m + i, c + j):
                    moves.append((m + i, c + j, 1))
    return moves

def bfs():
    start_state = (3, 3, 1)
    goal_state = (0, 0, 0)
    queue = deque([(start_state, [])])
    visited = set([start_state])
    
    while queue:
        (m, c, b), path = queue.popleft()
        
        if (m, c, b) == goal_state:
            return path
        
        for move in get_possible_moves(m, c, b):
            if move not in visited:
                visited.add(move)
                queue.append((move, path + [(m, c, b)]))
    
    return None

def print_solution(solution):
    if solution:
        print("Solution found!")
        for i, (m, c, b) in enumerate(solution):
            direction = "left to right" if b == 1 else "right to left"
            print(f"Step {i + 1}: Move {m} missionaries and {c} cannibals from {direction}.")
    else:
        print("No solution found.")

if __name__ == "__main__":
    solution = bfs()
    print_solution(solution)
