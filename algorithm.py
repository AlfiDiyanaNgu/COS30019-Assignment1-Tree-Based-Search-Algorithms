import sys
from collections import deque
import heapq  
from collections import defaultdict
import time
import tracemalloc
import itertools

#=========================================== Uninformed Search ===========================================

def bfs(start, goals, rows, cols, obstacles):
    queue = deque([(start, [start], [])])  # (current position, path, moves) & start with path[start]
    visited = set()
    nodes_explored = 0
    nodes_explored_list = []

    while queue:
        (x, y), path, moves = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y)) # mark node to as visited
        nodes_explored += 1
        nodes_explored_list.append((x,y)) # save order exploration

        # when goal found
        if (x, y) in goals:
            return (x, y), nodes_explored, moves, nodes_explored_list

        # Explore neighbors: Up, Left, Down, Right
        # FIFO
        directions = [(x, y-1, 'up'), (x-1, y, 'left'), (x, y+1, 'down'), (x+1, y, 'right')]
        # explore valid neighbors
        for mx, my, move in directions:
            next_pos = (mx, my)
            # validate node
            if (
                0 <= mx < cols and 
                0 <= my < rows and 
                next_pos not in obstacles and 
                next_pos not in visited
            ):
                queue.append(((mx, my), path + [next_pos], moves + [move]))
    # If queue is empty and no goal was found
    return None, nodes_explored, [], nodes_explored_list

def dfs(start, goals, rows, cols, obstacles):
    stack = deque([(start, [start], [])])  # (current position, path, moves)
    visited = set()
    nodes_explored = 0
    nodes_explored_list = []

    while stack:
        (x, y), path, moves = stack.pop() #LIFO

        if (x, y) in visited:
            continue
        visited.add((x, y))

        nodes_explored += 1
        nodes_explored_list.append((x,y))

        if (x, y) in goals:
            return (x, y), nodes_explored, moves, nodes_explored_list

        # Push directions in REVERSE so 'up' has the highest priority
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
    goals = list(goals)
    if not goals:
        return None, 0, [], []
    if start in goals:
        return start, 0, [], []

    obstacles = set(obstacles)
    move_reversal = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    directions = [(0, -1, 'up'), (-1, 0, 'left'), (0, 1, 'down'), (1, 0, 'right')]

    # forward queue
    forward_queue = deque([(start, [])])
    forward_visited = {start: (None, None)}

    # Stores visited nodes for each goal with trace info
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

        #pop from forward queue
        if current == 0 and forward_queue:
            (fx, fy), f_moves = forward_queue.popleft()
            nodes_explored += 1
            nodes_explored_list.append((fx, fy))

            # check if node already visited by any backward search
            for goal in goals:
                if (fx, fy) in backward_visited[goal]:
                    # Build path
                    forward_path = []
                    node = (fx, fy)
                    while forward_visited[node][0] is not None:
                        parent, move = forward_visited[node]
                        forward_path.append(move)
                        node = parent
                    forward_path.reverse() #reverse to get correct order

                    # reconstruct backward path from current node to goal
                    backward_path = []
                    node = (fx, fy)
                    while backward_visited[goal][node][0] is not None:
                        parent, move = backward_visited[goal][node]
                        backward_path.append(move_reversal[move])
                        node = parent

                    # forward + backward
                    return goal, nodes_explored, forward_path + backward_path, nodes_explored_list
            # expand neighbors if still not connect
            for dx, dy, move in directions:
                nx, ny = fx + dx, fy + dy
                next_pos = (nx, ny)
                if 0 <= nx < cols and 0 <= ny < rows and next_pos not in obstacles and next_pos not in forward_visited:
                    forward_visited[next_pos] = ((fx, fy), move)
                    forward_queue.append((next_pos, f_moves + [move]))

        # backward search
        elif current > 0:
            goal = goals[current - 1]  # Determine which goal is active in this round
            if backward_queues[goal]:
                (bx, by), b_moves = backward_queues[goal].popleft()
                nodes_explored += 1
                nodes_explored_list.append((bx, by))

                # Check if this node was already visited by forward search
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
                        # Mark node as visited by this goal
                        backward_visited[goal][next_pos] = ((bx, by), move)
                        backward_queues[goal].append((next_pos, b_moves + [move]))

        turn += 1  # rotate to next agent

    return None, nodes_explored, [], nodes_explored_list

#=========================================== Informed Search ===========================================

def gbfs(start, goals, rows, cols, obstacles):
    queue = []
    counter = 0  # For tie-breaking (same heuristics) using insertion order
    heapq.heappush(queue, (0, counter, start, [start], []))  # (heuristic, insertion_id, pos, path, moves)
    counter += 1
    visited = set()
    nodes_explored = 0
    nodes_explored_list = []

    while queue:
        _, _, (x, y), path, moves = heapq.heappop(queue)  # Pop the node with the lowest heuristic value (greedy)

        if (x, y) in visited:
            continue
        visited.add((x, y))

        nodes_explored += 1
        nodes_explored_list.append((x,y))

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
    # tie-breaking
    MOVE_PRIORITY = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
    directions = [
        (0, -1, 'up'),
        (-1, 0, 'left'),
        (0, 1, 'down'),
        (1, 0, 'right')
    ]

    queue = []
    g_scores = {start: 0} #store cost from start to each position
    visited = set()
    initial_h = heuristic(start, goals)
    heapq.heappush(queue, (initial_h, 0, start, []))
    nodes_explored = 0 # Number of expanded nodes
    nodes_explored_list = []  # Stores coordinates of all expanded nodes

    while queue:
        # pop lowest fn node
        f, g_cost, current, moves = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1
        nodes_explored_list.append((current))

        if current in goals:
            return current, nodes_explored, moves, nodes_explored_list

        # Explore neighboring positions in priority order
        for dx, dy, move in directions:
            nx, ny = current[0] + dx, current[1] + dy
            new_pos = (nx, ny)

            if 0 <= nx < cols and 0 <= ny < rows and new_pos not in obstacles:
                new_g = g_cost + 1 # Uniform cost for moving one step
                if new_pos in visited:
                    continue
                 # Update g-score if it's better or not seen before
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
    forward_visited = {start: (0, None, None)} # position: (g_cost, parent, move)
    insertion_counter = itertools.count() # Tie-breaker for heap
    heapq.heappush(forward_queue, (heuristic(start, goals), next(insertion_counter), 0, start))

    # Backward search (from goals to start), one queue per goal
    backward_queues = {}
    backward_visited = defaultdict(dict)
    
    # Initialize backward search for each goal
    for goal in goals:
        h = heuristic(goal, [start])
        backward_queues[goal] = [(h, next(insertion_counter), 0, goal)]
        backward_visited[goal][goal] = (0, None, None)

    nodes_explored = 0
    nodes_explored_list = []
    goal_cycle = itertools.cycle(goals) # round-robin processing of backward queues

    while forward_queue or any(backward_queues.values()):
        # Forward step
        if forward_queue:
            _, _, f_g, (fx, fy) = heapq.heappop(forward_queue)
            nodes_explored += 1
            nodes_explored_list.append((fx, fy))

            # Check for intersection with any backward search
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

                    # Reconstruct backward path
                    backward_path = []
                    node = (fx, fy)
                    while backward_visited[goal][node][1] is not None:
                        parent, move = backward_visited[goal][node][1:]
                        backward_path.append(move_reversal[move])
                        node = parent

                    return goal, nodes_explored, forward_path + backward_path, nodes_explored_list

             # Explore neighbors in forward direction
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

            # Check for intersection with forward search
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
            
            # Explore neighbors in backward direction
            for dx, dy, move in directions:
                nx, ny = bx + dx, by + dy
                next_pos = (nx, ny)
                if (0 <= nx < cols and 0 <= ny < rows and next_pos not in obstacles
                        and next_pos not in backward_visited[goal]
                        # Prevent this node from being added by multiple backward searches
                        and all(next_pos not in visited for g, visited in backward_visited.items() if g != goal)):
                    new_g = b_g + 1
                    h = heuristic(next_pos, [start])  # Backward search always targets start
                    backward_visited[goal][next_pos] = (new_g, (bx, by), move)
                    heapq.heappush(queue, (new_g + h, next(insertion_counter), new_g, next_pos))

    return None, nodes_explored, [], nodes_explored_list

def heuristic(pos, goals):
    x, y = pos
    return min(abs(x - gx) + abs(y - gy) for (gx, gy) in goals)

# this function measure runtime & memory usage
def run_with_metrics(search_function, *args, **kwargs):
    start_time = time.time()
    tracemalloc.start() # track memory

    #Calls the search algorithm and collects results.
    goal, nodes_explored, moves, nodes_explored_list = search_function(*args, **kwargs)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time Taken: {end_time - start_time:.6f} seconds")
    print(f"Total Memory Used (Peak): {peak / 1024:.6f} KB ({peak / (1024*1024):.6f} MB)\n")

    return goal, nodes_explored, moves, nodes_explored_list