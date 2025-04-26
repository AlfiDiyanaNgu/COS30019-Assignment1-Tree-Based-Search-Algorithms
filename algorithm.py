import sys
from collections import deque
import heapq  
from collections import defaultdict
import time
import tracemalloc
import itertools

#=========================================== Uninformed Search ===========================================

def bfs(start, goals, rows, cols, obstacles):
    queue = deque([(start, [start], [])])  # (current position, path, moves)
    visited = set()
    nodes_explored = 0
    nodes_explored_list = []

    while queue:
        (x, y), path, moves = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        nodes_explored += 1
        nodes_explored_list.append((x,y))

        if (x, y) in goals:
            return (x, y), nodes_explored, moves, nodes_explored_list

        # Explore neighbors: Up, Left, Down, Right
        # FIFO
        directions = [(x, y-1, 'up'), (x-1, y, 'left'), (x, y+1, 'down'), (x+1, y, 'right')]
        for mx, my, move in directions:
            next_pos = (mx, my)
            if (
                0 <= mx < cols and 
                0 <= my < rows and 
                next_pos not in obstacles and 
                next_pos not in visited
            ):
                queue.append(((mx, my), path + [next_pos], moves + [move]))

    return None, nodes_explored, [], nodes_explored_list

def dfs(start, goals, rows, cols, obstacles):
    """Depth-First Search Algorithm to find a path."""
    stack = deque([(start, [start], [])])  # (current position, path, moves)
    visited = set()
    nodes_explored = 0
    nodes_explored_list = []

    while stack:
        (x, y), path, moves = stack.pop()

        if (x, y) in visited:
            continue
        visited.add((x, y))

        nodes_explored += 1
        nodes_explored_list.append((x,y))

        # print(f"Expanded Node DFS: ({x}, {y})")  # <-- This prints each expanded node

        if (x, y) in goals:
            return (x, y), nodes_explored, moves, nodes_explored_list

        # Push directions in **reverse** so 'up' has the highest priority
        move_order = [
            (x+1, y, 'right'),   # lowest priority
            (x, y+1, 'down'),
            (x-1, y, 'left'),
            (x, y-1, 'up')       # highest priority
        ]

        for mx, my, move in move_order:
            next_pos = (mx, my)
            if (
                0 <= mx < cols and
                0 <= my < rows and
                next_pos not in obstacles and
                next_pos not in visited
            ):
                stack.append((next_pos, path + [next_pos], moves + [move]))

    return None, nodes_explored, [], nodes_explored_list

def bidirectional_bfs(start, goals, rows, cols, obstacles):
    """
    Bidirectional BFS with round-robin backward search per goal.
    Returns: (reached_goal, nodes_explored, path_moves, nodes_explored_list)
    """
    goals = list(goals)
    if not goals:
        return None, 0, [], []
    if start in goals:
        return start, 0, [], []

    obstacles = set(obstacles)
    move_reversal = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    directions = [(0, -1, 'up'), (-1, 0, 'left'), (0, 1, 'down'), (1, 0, 'right')]

    forward_queue = deque([(start, [])])
    forward_visited = {start: (None, None)}

    # Backward per goal
    backward_queues = {goal: deque([(goal, [])]) for goal in goals}
    backward_visited = {goal: {goal: (None, None)} for goal in goals}

    # Global visited by any goal to avoid overlap
    global_backward_visited = {goal for goal in goals}

    nodes_explored = 0
    nodes_explored_list = []

    turn = 0  # round-robin turn
    while forward_queue or any(backward_queues.values()):
        # Round-robin: start -> goalA -> goalB -> start -> ...
        current = turn % (len(goals) + 1)

        if current == 0 and forward_queue:
            (fx, fy), f_moves = forward_queue.popleft()
            nodes_explored += 1
            nodes_explored_list.append((fx, fy))
            #print(f"Expanded Node Forward Bi-BFS: ({fx}, {fy})")

            for goal in goals:
                if (fx, fy) in backward_visited[goal]:
                    # Build path
                    forward_path = []
                    node = (fx, fy)
                    while forward_visited[node][0] is not None:
                        parent, move = forward_visited[node]
                        forward_path.append(move)
                        node = parent
                    forward_path.reverse()

                    backward_path = []
                    node = (fx, fy)
                    while backward_visited[goal][node][0] is not None:
                        parent, move = backward_visited[goal][node]
                        backward_path.append(move_reversal[move])
                        node = parent

                    return goal, nodes_explored, forward_path + backward_path, nodes_explored_list

            for dx, dy, move in directions:
                nx, ny = fx + dx, fy + dy
                next_pos = (nx, ny)
                if 0 <= nx < cols and 0 <= ny < rows and next_pos not in obstacles and next_pos not in forward_visited:
                    forward_visited[next_pos] = ((fx, fy), move)
                    forward_queue.append((next_pos, f_moves + [move]))

        elif current > 0:
            goal = goals[current - 1]
            if backward_queues[goal]:
                (bx, by), b_moves = backward_queues[goal].popleft()
                nodes_explored += 1
                nodes_explored_list.append((bx, by))
                #print(f"Expanded Node Backward Bi-BFS (goal {goal}): ({bx}, {by})")

                if (bx, by) in forward_visited:
                    # Build path
                    forward_path = []
                    node = (bx, by)
                    while forward_visited[node][0] is not None:
                        parent, move = forward_visited[node]
                        forward_path.append(move)
                        node = parent
                    forward_path.reverse()

                    backward_path = []
                    node = (bx, by)
                    while backward_visited[goal][node][0] is not None:
                        parent, move = backward_visited[goal][node]
                        backward_path.append(move_reversal[move])
                        node = parent

                    return goal, nodes_explored, forward_path + backward_path, nodes_explored_list

                for dx, dy, move in directions:
                    nx, ny = bx + dx, by + dy
                    next_pos = (nx, ny)

                    # Make sure this node is:
                    # - Not an obstacle
                    # - Not already visited by this goal
                    # - Not already visited by other goals
                    if (
                        0 <= nx < cols and 0 <= ny < rows and
                        next_pos not in obstacles and
                        next_pos not in backward_visited[goal] and
                        all(next_pos not in backward_visited[other] for other in goals if other != goal)
                    ):
                        backward_visited[goal][next_pos] = ((bx, by), move)
                        backward_queues[goal].append((next_pos, b_moves + [move]))

        turn += 1  # rotate to next agent

    return None, nodes_explored, [], nodes_explored_list

#=========================================== Informed Search ===========================================
def gbfs(start, goals, rows, cols, obstacles):
    """Greedy Best-First Search with strict insertion-order tie-breaking and resource tracking."""
    queue = []
    counter = 0  # For tie-breaking using insertion order
    heapq.heappush(queue, (0, counter, start, [start], []))  # (heuristic, insertion_id, pos, path, moves)
    counter += 1
    visited = set()
    nodes_explored = 0
    nodes_explored_list = []

    while queue:
        _, _, (x, y), path, moves = heapq.heappop(queue)

        if (x, y) in visited:
            continue
        visited.add((x, y))

        nodes_explored += 1
        nodes_explored_list.append((x,y))

        # print(f"Expanded Node GBFS: ({x}, {y})")  # <-- This prints each expanded node

        if (x, y) in goals:
            return (x, y), nodes_explored, moves, nodes_explored_list

        # Move order: up, left, down, right
        moves_priority = [
            (x, y-1, 'up'),
            (x-1, y, 'left'),
            (x, y+1, 'down'),
            (x+1, y, 'right')
        ]

        for mx, my, move in moves_priority:
            if (0 <= mx < cols and 0 <= my < rows and 
                (mx, my) not in obstacles and 
                (mx, my) not in visited):
                
                heuristic = min(abs(mx - gx) + abs(my - gy) for (gx, gy) in goals)
                heapq.heappush(queue, (heuristic, counter, (mx, my), path + [(mx, my)], moves + [move]))
                counter += 1

    # If no path is found
    return None, nodes_explored, [], nodes_explored_list

def astar(start, goals, rows, cols, obstacles):
    """
    A* Search matching class-based implementation logic.
    Priority queue: (f(n), g(n), pos, moves)
    """
    MOVE_PRIORITY = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
    directions = [
        (0, -1, 'up'),
        (-1, 0, 'left'),
        (0, 1, 'down'),
        (1, 0, 'right')
    ]

    def heuristic(pos, goals):
        return min(abs(pos[0] - gx) + abs(pos[1] - gy) for gx, gy in goals)

    queue = []
    g_scores = {start: 0}
    visited = set()
    initial_h = heuristic(start, goals)
    heapq.heappush(queue, (initial_h, 0, start, []))
    nodes_explored = 0
    nodes_explored_list = []

    while queue:
        f, g_cost, current, moves = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1
        nodes_explored_list.append((current))
        # print(f"Path: {moves}")

        if current in goals:
            return current, nodes_explored, moves, nodes_explored_list

        for dx, dy, move in directions:
            nx, ny = current[0] + dx, current[1] + dy
            new_pos = (nx, ny)

            if 0 <= nx < cols and 0 <= ny < rows and new_pos not in obstacles:
                new_g = g_cost + 1
                if new_pos in visited:
                    continue
                if new_pos not in g_scores or new_g < g_scores[new_pos]:
                    g_scores[new_pos] = new_g
                    new_h = heuristic(new_pos, goals)
                    new_f = new_g + new_h
                    heapq.heappush(queue, (new_f, new_g, new_pos, moves + [move]))

    return None, nodes_explored, [], nodes_explored_list

def bidirectional_astar(start, goals, rows, cols, obstacles):

    goals = list(goals)
    if not goals:
        return None, 0, [], []
    if start in goals:
        return start, 0, [], []

    obstacles = set(obstacles)
    move_reversal = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    directions = [(0, -1, 'up'), (-1, 0, 'left'), (0, 1, 'down'), (1, 0, 'right')]

    forward_queue = []
    forward_visited = {start: (0, None, None)}
    insertion_counter = itertools.count()
    heapq.heappush(forward_queue, (heuristic(start, goals), next(insertion_counter), 0, start))

    backward_queues = {}
    backward_visited = defaultdict(dict)

    for goal in goals:
        h = heuristic(goal, [start])
        backward_queues[goal] = [(h, next(insertion_counter), 0, goal)]
        backward_visited[goal][goal] = (0, None, None)

    nodes_explored = 0
    nodes_explored_list = []
    goal_cycle = itertools.cycle(goals)

    while forward_queue or any(backward_queues.values()):
        # Forward step
        if forward_queue:
            _, _, f_g, (fx, fy) = heapq.heappop(forward_queue)
            nodes_explored += 1
            nodes_explored_list.append((fx, fy))

            for goal in goals:
                if (fx, fy) in backward_visited[goal]:
                    # Reconstruct path
                    forward_path = []
                    node = (fx, fy)
                    while forward_visited[node][1] is not None:
                        parent, move = forward_visited[node][1:]
                        forward_path.append(move)
                        node = parent
                    forward_path.reverse()

                    backward_path = []
                    node = (fx, fy)
                    while backward_visited[goal][node][1] is not None:
                        parent, move = backward_visited[goal][node][1:]
                        backward_path.append(move_reversal[move])
                        node = parent

                    return goal, nodes_explored, forward_path + backward_path, nodes_explored_list

            for dx, dy, move in directions:
                nx, ny = fx + dx, fy + dy
                next_pos = (nx, ny)
                if 0 <= nx < cols and 0 <= ny < rows and next_pos not in obstacles:
                    new_g = f_g + 1
                    if next_pos not in forward_visited or new_g < forward_visited[next_pos][0]:
                        h = heuristic(next_pos, goals)
                        forward_visited[next_pos] = (new_g, (fx, fy), move)
                        heapq.heappush(forward_queue, (new_g + h, next(insertion_counter), new_g, next_pos))

        # Backward step (round robin)
        for _ in range(len(goals)):
            goal = next(goal_cycle)
            queue = backward_queues[goal]
            if not queue:
                continue

            _, _, b_g, (bx, by) = heapq.heappop(queue)
            nodes_explored += 1
            nodes_explored_list.append((bx, by))

            if (bx, by) in forward_visited:
                forward_path = []
                node = (bx, by)
                while forward_visited[node][1] is not None:
                    parent, move = forward_visited[node][1:]
                    forward_path.append(move)
                    node = parent
                forward_path.reverse()

                backward_path = []
                node = (bx, by)
                while backward_visited[goal][node][1] is not None:
                    parent, move = backward_visited[goal][node][1:]
                    backward_path.append(move_reversal[move])
                    node = parent

                return goal, nodes_explored, forward_path + backward_path, nodes_explored_list

            for dx, dy, move in directions:
                nx, ny = bx + dx, by + dy
                next_pos = (nx, ny)
                if (0 <= nx < cols and 0 <= ny < rows and next_pos not in obstacles
                        and next_pos not in backward_visited[goal]
                        and all(next_pos not in visited for g, visited in backward_visited.items() if g != goal)):
                    new_g = b_g + 1
                    h = heuristic(next_pos, [start])
                    backward_visited[goal][next_pos] = (new_g, (bx, by), move)
                    heapq.heappush(queue, (new_g + h, next(insertion_counter), new_g, next_pos))

    return None, nodes_explored, [], nodes_explored_list

def heuristic(pos, goals):
    """Manhattan distance to the nearest goal."""
    x, y = pos
    return min(abs(x - gx) + abs(y - gy) for (gx, gy) in goals)

def run_with_metrics(search_function, *args, **kwargs):
    start_time = time.time()
    tracemalloc.start()

    goal, nodes_explored, moves, nodes_explored_list = search_function(*args, **kwargs)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time Taken: {end_time - start_time:.6f} seconds")
    print(f"Total Memory Used (Peak): {peak / 1024:.6f} KB ({peak / (1024*1024):.6f} MB)\n")

    return goal, nodes_explored, moves, nodes_explored_list