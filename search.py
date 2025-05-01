import sys
import time
import tracemalloc
from algorithm import bfs, dfs, bidirectional_bfs, astar, gbfs, bidirectional_astar
from algorithm import run_with_metrics
import tkinter as tk

CELL_SIZE = 30
DELAY = 0.05  # change the speed of animation here

def display_grid(file_path, path=None, show=True):
    with open(file_path, 'r') as file:

        #first line - grid size
        grid_line = file.readline().strip() # strip() = remove whitespace including \n
        grid_dimensions = grid_line.strip('[]').split(',') # remove [] and ,
        rows, cols = int(grid_dimensions[0].strip()), int(grid_dimensions[1].strip()) #convert str to int

        #sec line - initial state
        initial_state_coordinates = file.readline().strip('()\n ').split(',')
        col_idx, row_idx = int(initial_state_coordinates[0].strip()), int(initial_state_coordinates[1].strip())
        start = (col_idx, row_idx)

        #third line - goal state
        goal_coordinates = file.readline().strip().split('|')
        goals = set()
        for goal in goal_coordinates: #loop goal coordinate bcs got more than 1 goal
            goal_col, goal_row = map(int, goal.strip('() ').split(',')) #map() convert to int
            goals.add((goal_col, goal_row))

        #start from fourth line - obstacles 
        obstacles = set()
        for line in file: 
            if line.strip():
                obstacle_values = list(map(int, line.strip('()\n ').split(','))) #list() convert map into list 
                obstacles_col, obstacles_row = obstacle_values[0], obstacle_values[1]
                width, height = obstacle_values[2], obstacle_values[3]
                for i in range(width):
                    for j in range(height):
                        obstacles.add((obstacles_col + i, obstacles_row + j))

    if show:
        print(f"\nGrid Dimensions: {rows} rows, {cols} columns")
        print(f"Initial State: {start}")
        print(f"Goal States: {goals}")
        print(f"Obstacles: {obstacles}")
        print("\nGenerated Grid:")

        # Check current position to display grid
        for row in range(rows):
            row_display = []
            for col in range(cols):
                pos = (col, row) #current coordinate being checked
                if pos == start:
                    row_display.append('X')
                elif pos in goals:
                    row_display.append('G')
                elif pos in obstacles:
                    row_display.append('#')
                elif path and pos in path and pos != start and pos not in goals:
                    row_display.append('P')
                else:
                    row_display.append('.')
            print(' '.join(row_display))

    return start, goals, rows, cols, obstacles

def draw_grid_with_animation(rows, cols, start, goals, obstacles, path_coords, nodes_explored_list, algorithm_name):
    root = tk.Tk()
    root.title(f"Robot Navigation - {algorithm_name}")

    # Set window size and padding
    root.geometry(f"{cols * CELL_SIZE + 300}x{rows * CELL_SIZE + 300}")
    
    # Create a label to display the algorithm name
    label = tk.Label(root, text=algorithm_name, font=("Georgia", 25, "bold"), fg="black")
    label.pack(pady=10)

    #Create canvas widget (blank place to draw graphics within a window)
    canvas = tk.Canvas(root, width=cols * CELL_SIZE, height=rows * CELL_SIZE)
    canvas.pack(pady=10)

    # Create 2D list
    rects = [[None for _ in range(cols)] for _ in range(rows)]

    # Create grid
    for row in range(rows):
        for col in range(cols):
            color = "white"
            if (col, row) in obstacles:
                color = "black"
            elif (col, row) == start:
                color = "red"
            elif (col, row) in goals:
                color = "green"

            rect = canvas.create_rectangle(
                col * CELL_SIZE, row * CELL_SIZE,
                (col + 1) * CELL_SIZE, (row + 1) * CELL_SIZE,
                fill=color, outline="gray"
            )
            rects[row][col] = rect

    def animate():
        try: # handle if user close animation
            # Animate explored nodes
            for idx, (col, row) in enumerate(nodes_explored_list, start=1):
                if (col, row) != start and (col, row) not in goals:
                    canvas.itemconfig(rects[row][col], fill="blue")
                    canvas.create_text( # draw order of node
                        col * CELL_SIZE + CELL_SIZE // 2,
                        row * CELL_SIZE + CELL_SIZE // 2,
                        text=str(idx),
                        font=("Helvetica", 10, "bold"),
                        fill="white"
                    )
                    root.update() #refresh GUI to show color
                    time.sleep(DELAY) # make animation happen slowly

            # Animate path if it exists
            if path_coords:
                for col, row in path_coords:
                    if (col, row) != start and (col, row) not in goals:
                        canvas.itemconfig(rects[row][col], fill="yellow")
                        root.update()
                        time.sleep(DELAY)
            else:
                print("No path found — skipping path animation.")
        except tk.TclError: #handle when window is closed while animating
            return  # Stop animation if window is closed

    root.after(100, animate)
    root.mainloop() #handle event loop

def main():
    if len(sys.argv) < 3:
        print("Usage: python search.py <input_file> <algorithm>")
        sys.exit(1)

    input_file = sys.argv[1]
    algorithm = sys.argv[2].lower()

    try:
        # Load input, but do not display yet
        start, goals, rows, cols, obstacles = display_grid(input_file, show=False)  

        #use for argument
        algorithm_functions = {
            "bfs": bfs,
            "dfs": dfs,
            "gbfs": gbfs,
            "as": astar,
            "cus1": bidirectional_bfs,
            "cus2": bidirectional_astar
        }

        #use for display algorithm name at window 
        algorithm_display_names = {
            "bfs": "Breadth-First Search",
            "dfs": "Depth-First Search",
            "gbfs": "Greedy Best-First Search",
            "as": "A* Search",
            "cus1": "Bidirectional BFS",
            "cus2": "Bidirectional A*"
        }

        # get algorithm name for GUI display
        algorithm_title = algorithm_display_names.get(algorithm, algorithm.upper())

        if algorithm in algorithm_functions:
            goal, nodes_explored, path, nodes_explored_list = run_with_metrics(algorithm_functions[algorithm], start, goals, rows, cols, obstacles)
        else:
            print("Unsupported algorithm, Choose other search algorithm: \nbfs\ndfs\ncus1\ngbfs\nas\ncus2")
            sys.exit(1)

        print(f"{input_file} {algorithm}")
        if goal:
            print(f"<Node {goal}> {nodes_explored}")
            print(path)

            # Reconstruct full path coordinates from move directions
            full_path = [start]
            x, y = start
            for move in path:
                if move == "up":
                    y -= 1
                elif move == "down":
                    y += 1
                elif move == "left":
                    x -= 1
                elif move == "right":
                    x += 1
                full_path.append((x, y))

            print(f"Step:  {len(path)}")

            # Display grid with path
            display_grid(input_file, path=full_path)

            # Call animation with explored nodes and path coordinates
            draw_grid_with_animation(rows, cols, start, goals, obstacles, full_path, nodes_explored_list, algorithm_name=algorithm_title)


        else:
            print(f"No goal is reachable; {nodes_explored}")
            display_grid(input_file, path=None)
            # Call animation with explored nodes and path coordinates
            full_path = None
            draw_grid_with_animation(rows, cols, start, goals, obstacles, full_path, nodes_explored_list, algorithm_name=algorithm_title)


    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except ValueError as e:
        print(f"Error: Unable to parse grid dimensions or coordinates. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()