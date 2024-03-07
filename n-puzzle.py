# Nhóm 11
# Đề tài: Thiết kế và Xây Dựng Game N-Puzzle

# Thành viên nhóm:
#   Nguyễn Khánh Quy - 21110282 (Nhóm trưởng)
#   Võ Chí Khương - 21110221
#   Nguyễn Hồng Thông Điệp - 21110166

import os
import sys
import time
import queue
import heapq
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque
from PIL import Image, ImageTk

global ROW, COL, image_mapping

total_steps = 0
step_count = 0
solving_time = 0
total_nodes = 0
depth_limit = 1
speed = 0.2
buttons = []

def center_window(root, width, height):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    x_coordinate = (screen_width - width) // 2
    y_coordinate = (screen_height - 100 - height) // 2
    
    root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")


def prepare_puzzle():
    global ROW, COL
    ROW = int(row_combobox.get())
    COL = int(col_combobox.get())
    menu_window.destroy()
    main_game(ROW, COL)


def main_game(ROW, COL):
    global puzzle, saved_state, goal, SIZE, game_window
    SIZE = int(300 / ROW)
    puzzle = list(range(0, ROW * COL))
    saved_state = list(puzzle)
    goal = list(range(0, ROW * COL))

    def create_square_image():
        square_image = Image.new("RGB", (100, 100), "#0666a1")
        return square_image

    def is_solved(puzzle):
        return puzzle == goal
    
    def update_display():
        stock = ROW * COL + 1
        for i in range(ROW):
            for j in range(COL):
                button = buttons[i][j]
                value = puzzle[i * COL + j]
                photo = image_mapping[value]

                if value == 0:
                    button.config(text="", image=photo)
                else:
                    if any(f"pyimage{k}" == str(photo) for k in range(1, stock)):
                        button.config(
                            text=value,
                            compound="center",
                            fg="white",
                            font=("Tahoma", 20, "bold"),
                            image=photo,
                        )
                    else:
                        button.config(
                            text="",
                            compound="center",
                            fg="white",
                            font=("Tahoma", 20, "bold"),
                            image=photo,
                        )
                button.config(state=tk.NORMAL if value else tk.DISABLED)

    def move(puzzle, move_to):
        global step_count
        empty_index = puzzle.index(0)
        move_index = puzzle.index(move_to)

        if (
            empty_index % COL == move_index % COL
            and abs(empty_index - move_index) == COL
        ) or (
            empty_index // COL == move_index // COL
            and abs(empty_index - move_index) == 1
        ):
            puzzle[empty_index] = move_to
            puzzle[move_index] = 0

            update_display()
            update_step_count(step_count)


    def on_button_click(row, col):
        move_to = puzzle[row * COL + col]
        global step_count
        step_count += 1
        move(puzzle, move_to)
        
    def on_key_press(event):
        zero_index = puzzle.index(0)
        row, col = divmod(zero_index, COL)

        if event.keysym == "w" and row > 0: # Up
            move_to = puzzle[zero_index - COL]
        elif event.keysym == "s" and row < ROW - 1: # Down
            move_to = puzzle[zero_index + COL]
        elif event.keysym == "a" and col > 0: # Left
            move_to = puzzle[zero_index - 1]
        elif event.keysym == "d" and col < COL - 1: # Right
            move_to = puzzle[zero_index + 1]
        else:
            return

        global step_count
        step_count += 1
        move(puzzle, move_to)

    def possible_moves(current_node):
        moves = []

        empty_index = current_node.index(0)
        row, col = empty_index // COL, empty_index % COL

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < ROW and 0 <= new_col < COL:
                new_empty_index = new_row * COL + new_col
                new_node = list(current_node)
                new_node[empty_index], new_node[new_empty_index] = (
                    new_node[new_empty_index],
                    new_node[empty_index],
                )
                moves.append((new_node, new_node[empty_index]))
        random.shuffle(moves)
        return moves

    def random_shuffle(puzzle):
        visited = set()
        visited.add(tuple(puzzle))
        count = 0
        while count < 100:
            nodes = possible_moves(puzzle)
            for item in nodes:
                node, pos_move = item
                if tuple(node) not in visited:
                    move(puzzle, pos_move)
                    visited.add(tuple(node))
            count += 1

    def state_to_string(state):
        puzzle_string = ""
        for i in range(ROW):
            for j in range(COL):
                cell_value = str(state[i * COL + j]).zfill(1)
                puzzle_string += cell_value
        return puzzle_string

    def update_infor_lables():
        update_total_nodes_count(total_nodes)
        update_step_count(step_count)
        update_depth_limit_count(depth_limit)
        update_total_steps_count(total_steps)
        update_solving_time(solving_time)
        update_state_lable(state_to_string(saved_state), state_to_string(goal))

    def reset_infor_lables():
        global total_nodes, step_count, total_steps, solving_time, depth_limit
        step_count = 0
        total_steps = 0
        total_nodes = 0
        solving_time = 0
        update_infor_lables()

    def update_depth_limit_count(count):
        depth_limit_label.config(text=f"{count}")

    def update_step_count(count):
        step_label.config(text=f"{count}")

    def update_total_steps_count(count):
        total_steps_label.config(text=f"{count}")

    def update_total_nodes_count(count):
        total_nodes_label.config(text=f"{count}")

    def update_solving_time(count):
        time_label.config(text=f"{count:.2f}s")

    def update_state_lable(str_saved_state, str_goal):
        if ROW > 3 or COL > 3:
            game_label.config(text="Trạng thái xuất phát: " + str_saved_state + "\n    -> " + "Trạng thái đích: " + str_goal)
        else:
            game_label.config(text="Trạng thái xuất phát: " + str_saved_state + " -> " + "Trạng thái đích: " + str_goal)

    def bfs_solve(puzzle):
        global total_nodes
        total_nodes = 0
        visited = set()
        start_node = tuple(puzzle)
        queue = deque([(start_node, [])])
        while queue:
            current_node, path = queue.popleft()
            for item in possible_moves(current_node):
                node, pos_move = item
                if tuple(node) not in visited:
                    visited.add(tuple(node))
                    total_nodes += 1
                    new_path = path + [pos_move]
                    if is_solved(list(node)):
                        return new_path
                    queue.append((node, new_path))
        return None

    def dfs_solve(puzzle):
        global total_nodes
        total_nodes = 0
        start_node = tuple(puzzle)
        stack = [(start_node, [])]
        visited = set()
        while stack:
            current_node, path = stack.pop()
            for item in possible_moves(current_node):
                node, pos_move = item
                if tuple(node) not in visited:
                    total_nodes += 1
                    visited.add(tuple(node))
                    new_path = path + [pos_move]
                    if is_solved(list(node)):
                        return new_path
                    stack.append((node, new_path))
        return None

    def dls_solve(puzzle, depth_limit):
        global total_nodes
        start_node = tuple(puzzle)
        stack = [(start_node, [], 0)]
        visited = set()
        while stack:
            current_node, path, depth = stack.pop()
            if is_solved(list(current_node)):
                return path
            if depth == depth_limit:
                continue
            for item in possible_moves(current_node):
                node, pos_move = item
                if tuple(node) not in visited:
                    total_nodes += 1
                    visited.add(tuple(node))
                    new_path = path + [pos_move]
                    stack.append((node, new_path, depth + 1))
        return None

    def id_solve(puzzle):
        global depth_limit
        depth_limit = int(spinbox.get())
        result = dls_solve(puzzle, depth_limit)
        while not result:
            depth_limit += int(spinbox.get())
            result = dls_solve(puzzle, depth_limit)
        return result

    def ucs_solve(puzzle):
        global total_nodes
        total_nodes = 0
        priority_queue = queue.PriorityQueue()
        visited = set()
        start_node = tuple(puzzle)
        priority_queue.put((0, start_node, []))
        while not priority_queue.empty():
            cost, current_node, path = priority_queue.get()
            for item in possible_moves(current_node):
                node, pos_move = item
                if tuple(node) not in visited:
                    total_nodes += 1
                    new_path = path + [pos_move]
                    visited.add(tuple(node))
                    new_cost = cost + 1
                    if is_solved(list(node)):
                        return new_path
                    priority_queue.put((new_cost, tuple(node), new_path))
        return None

    def manhattan_distance(puzzle):
        distance = 0
        for i in range(ROW):
            for j in range(COL):
                if puzzle[i * COL + j] != 0:
                    correct_row = (puzzle[i * COL + j]) // COL
                    correct_col = (puzzle[i * COL + j]) % COL
                    distance += abs(i - correct_row) + abs(j - correct_col)
        return distance

    def hamming_distance(puzzle):
        distance = 0
        for i in range(ROW):
            for j in range(COL):
                if puzzle[i * COL + j] != 0 and puzzle[i * COL + j] != i * COL + j + 1:
                    distance += 1
        return distance

    def linear_conflict(puzzle):
        conflicts = 0
        for i in range(ROW):
            for j in range(COL):
                if puzzle[i * COL + j] != 0:
                    correct_row = (puzzle[i * COL + j] - 1) // COL
                    correct_col = (puzzle[i * COL + j] - 1) % COL
                    if i == correct_row and j != correct_col:
                        conflicts += sum(
                            1
                            for k in range(j + 1, COL)
                            if puzzle[i * COL + k] != 0
                            and (puzzle[i * COL + k] - 1) // COL == i
                            and (puzzle[i * COL + k] - 1) % COL < correct_col
                        )
                    elif j == correct_col and i != correct_row:
                        conflicts += sum(
                            1
                            for k in range(i + 1, ROW)
                            if puzzle[k * COL + j] != 0
                            and (puzzle[k * COL + j] - 1) % COL == j
                            and (puzzle[k * COL + j] - 1) // COL < correct_row
                        )
        return conflicts * 2

    def comparator(puzzle):
        if heuristic_rb.get() == "manhattan":
            return manhattan_distance(puzzle)
        elif heuristic_rb.get() == "hamming":
            return hamming_distance(puzzle)
        else:
            return linear_conflict(puzzle)

    def greedy_solve(puzzle):
        global total_nodes
        total_nodes = 0
        priority_queue = queue.PriorityQueue()
        visited = set()
        start_node = tuple(puzzle)
        priority_queue.put((0, start_node, []))
        while not priority_queue.empty():
            _, current_node, path = priority_queue.get()
            for item in possible_moves(current_node):
                node, pos_move = item
                if tuple(node) not in visited:
                    visited.add(tuple(node))
                    total_nodes += 1
                    new_path = path + [pos_move]
                    new_cost = comparator(node)
                    if is_solved(list(node)):
                        return new_path
                    priority_queue.put((new_cost, tuple(node), new_path))
        return None

    def A_solve(puzzle):
        global total_nodes
        total_nodes = 0
        priority_queue = [(comparator(puzzle), 0, tuple(puzzle), [])]
        visited = set()
        while priority_queue:
            _, g_value, current_node, path = heapq.heappop(priority_queue)
            for item in possible_moves(current_node):
                node, pos_move = item
                if tuple(node) not in visited:
                    visited.add(tuple(node))
                    total_nodes += 1
                    new_path = path + [pos_move]
                    new_cost = g_value + 1 + comparator(node)
                    if is_solved(list(node)):
                        return new_path
                    heapq.heappush(
                        priority_queue,
                        (
                            new_cost,
                            g_value + 1,
                            tuple(node),
                            new_path,
                        ),
                    )
        return None

    def IDA_solve(puzzle):
        global total_nodes
        total_nodes = 0
        threshold = comparator(puzzle)

        def dls(puzzle, threshold):
            global total_nodes
            start_node = tuple(puzzle)
            stack = [(start_node, [], 0)]
            visited = set()
            min_cost = float("inf")

            while stack:
                current_node, path, g_value = stack.pop()
                f_value = g_value + comparator(current_node)

                if f_value > threshold:
                    min_cost = min(min_cost, f_value)
                    continue

                if is_solved(list(current_node)):
                    return path, float("inf")

                for item in possible_moves(current_node):
                    node, pos_move = item
                    if tuple(node) not in visited:
                        total_nodes += 1
                        visited.add(tuple(node))
                        new_path = path + [pos_move]
                        stack.append((node, new_path, g_value + 1))

            return None, min_cost

        while True:
            result, new_threshold = dls(puzzle, threshold)
            if result is not None:
                return result
            if new_threshold == float("inf"):
                return None
            threshold = new_threshold
    
    def hc_solve(puzzle):
        global total_nodes
        total_nodes = 0
        start_node = tuple(puzzle)
        queue = deque([(start_node, [])])
        while queue:
            current_node, path = queue.popleft()
            node, pos_move = possible_moves(current_node)[0]
            cost = comparator(node)
            for i in range(1, len(possible_moves(current_node))):
                if cost >= comparator(node):
                    cost = comparator(node)
                    node, pos_move = possible_moves(current_node)[i]
                    total_nodes += 1
                else:
                    return None
            new_path = path + [pos_move]
            if is_solved(list(node)):
                return new_path
            queue.append((node, new_path))
        return None

    def beam_solve(puzzle):
        global total_nodes
        total_nodes = 0
        visited = set()
        start_node = tuple(puzzle)
        queue1 = deque([(start_node, [])])
        while queue1:
            current_node, path = queue1.popleft()
            k = random.randint(2, len(possible_moves(current_node)))
            top_k_elements = []
            priority_queue = queue.PriorityQueue()
            for item in possible_moves(current_node):
                node, pos_move = item
                priority_queue.put((comparator(node), node, pos_move))
            for _ in range(k):
                if not priority_queue.empty():
                    top_k_elements.append(priority_queue.get())
            for item in top_k_elements:
                _, node, pos_move = item
                if tuple(node) not in visited:
                    visited.add(tuple(node))
                    total_nodes += 1
                    new_path = path + [pos_move]
                    if is_solved(list(node)):
                        return new_path
                    queue1.append((node, new_path))
        return None
    
    def add_history(str_state, str_goal):
        if is_solved(puzzle):
            file_path = 'History/history.txt'
            if not os.path.exists(file_path):
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(' Trạng thái xuất phát: ' + str_state + '\n Trạng thái đích: ' + str_goal + '\n' + '-'*90 + '\n')
                    file.write(f"{' Thuật toán':<15} {'Tổng số bước':>17} {'Thời gian':>17} {'Tổng số đỉnh đã duyệt':>26} {'Độ sâu':>10}\n" + '-'*90 + '\n')
            with open(file_path, 'a') as file:
                file.write(f"{' ' + algorithm_combobox.get():<15} {str(total_steps):>17} {f"{solving_time:.2f}s":>17} {str(total_nodes):>26} {str(depth_limit) if algorithm_combobox.get() in ["DLS", "ID"] else "-":>10}" + "\n")
    
    def add_solution(str_state, str_goal, str_solution_1, str_solution_2):
        if not os.path.exists('Solution/solution_1.txt'):
                with open('Solution/solution_1.txt', 'a', encoding='utf-8') as file:
                    file.write(' Trạng thái xuất phát: ' + str_state + '\n Trạng thái đích: ' + str_goal + '\n' + '-'*42 + '\n')
        with open('Solution/solution_1.txt', 'a') as file:
                file.write(str_solution_1)
                
        if not os.path.exists('Solution/solution_2.txt'):
                with open('Solution/solution_2.txt', 'a', encoding='utf-8') as file:
                    file.write(' Trạng thái xuất phát: ' + str_state + '\n Trạng thái đích: ' + str_goal + '\n' + '-'*42 + '\n')
        with open('Solution/solution_2.txt', 'a') as file:
                file.write(str_solution_2)

    def disable_controls():
        widgets_to_disable = [
            spinbox,
            hamming_rb,
            manhattan_rb,
            linear_conflict_rb,
            algorithm_combobox,
        ]

        for widget in widgets_to_disable:
            widget.config(state=tk.DISABLED)

        for button in control_buttons:
            button.config(state=tk.DISABLED)

    def enable_controls():
        widgets_to_disable = [
            spinbox,
            hamming_rb,
            manhattan_rb,
            linear_conflict_rb,
            algorithm_combobox,
        ]

        exit_btn.config(state=tk.NORMAL)

        for widget in widgets_to_disable:
            widget.config(state=tk.NORMAL)
        algorithm_combobox.state(["readonly"])

        for button in control_buttons:
            button.config(state=tk.NORMAL)

    def validate_depth():
        try:
            value = int(spinbox.get())
            if 1 <= value <= 1000:
                pass
            else:
                if value < 1:
                    spinbox.delete(0, tk.END)
                    spinbox.insert(0, "1")
                else:
                    spinbox.delete(0, tk.END)
                    spinbox.insert(0, "1000")
        except ValueError:
            spinbox.delete(0, tk.END)
            spinbox.insert(0, "1")

    def on_spinbox_change():
        global depth_limit
        validate_depth()
        depth_limit = int(spinbox.get())
        update_infor_lables()

    def on_combobox_change(event):
        global depth_limit
        selected_value = algorithm_combobox.get()
        btn_reset_click()
        reset_infor_lables()
        update_solving_time(solving_time)

        depth_limit_header_label.grid_forget()
        depth_limit_label.grid_forget()
        optional_label.grid_forget()
        spinbox.grid_forget()
        hamming_rb.grid_forget()
        manhattan_rb.grid_forget()
        linear_conflict_rb.grid_forget()

        if selected_value == "ID":
            depth_limit = 1
            # depth_limit_header_label.grid(row=1, column=4, padx=20, pady=5)
            # depth_limit_label.grid(row=2, column=4, pady=10)
            optional_label.grid(row=6, column=1, padx=5, pady=5)
            optional_label.config(text="Độ sâu dần:")
            spinbox.grid(row=6, column=2, padx=5, pady=5)

        elif selected_value == "DLS":
            validate_depth()
            depth_limit = int(spinbox.get())
            # depth_limit_header_label.grid(row=1, column=4, padx=20, pady=5)
            # depth_limit_label.grid(row=2, column=4, pady=10)
            optional_label.grid(row=6, column=1, padx=5, pady=5)
            optional_label.config(text="Giới hạn độ sâu:")
            spinbox.grid(row=6, column=2, padx=5, pady=5)
        # elif (
        #     selected_value == "A-Star"
        #     or selected_value == "IDA-Star"
        #     or selected_value == "Greedy"
        # ):
        #     # hamming_rb.grid(row=3, column=3, padx=5, pady=5)
        #     # manhattan_rb.grid(row=3, column=2, padx=5, pady=5)
        #     # linear_conflict_rb.grid(row=3, column=4, padx=5, pady=5, columnspan=2)
        #     optional_label.grid(row=6, column=1, padx=5, pady=5)
        #     optional_label.config(text="Heuristic:")
        #     heuristic_combobox.grid(row=6, column=2, padx=5, pady=5)
            
        update_infor_lables()

    def run_algorithm():
        global stop_event, thread_count, solving_time, step_count, total_steps, speed, puzzle
        speed = 0.2
        reset_infor_lables()
        # update_solving_time(solving_time)
        disable_controls()
        control_buttons[0]['text'] = "Đang giải"
        algorithms = {
            "BFS": bfs_solve,
            "DFS": dfs_solve,
            "DLS": lambda puzzle: dls_solve(puzzle, depth_limit),
            "ID": id_solve,
            "UCS": ucs_solve,
            "Greedy": greedy_solve,
            "A-Star": A_solve,
            "IDA-Star": IDA_solve,
            "Hill Climbing": hc_solve,
            "Beam Search": beam_solve,
        }

        # Check folder
        if not os.path.exists('Data'):
            os.makedirs('Data')
        if not os.path.exists('History'):
            os.makedirs('History')
        if not os.path.exists('Solution'):
            os.makedirs('Solution')

        selected_algorithm = algorithm_combobox.get()
        if not os.path.exists('Data/' + selected_algorithm):
            os.makedirs('Data/' + selected_algorithm)
        
        str_state = state_to_string(saved_state)
        str_goal = state_to_string(goal)

        # Check solution
        if not os.path.exists('Data/' + selected_algorithm + '/' + str_state + '-' + str_goal + '.txt'):
            start_time = time.time()
            solution = algorithms[selected_algorithm](puzzle)
            solving_time = time.time() - start_time

            if stop_event:
                stop_event.set()
                thread_count.join()

            update_solving_time(solving_time)
            update_infor_lables()

            if solution:
                if (selected_algorithm != "ID") and (selected_algorithm != "DLS"):
                    with open('Data/' + selected_algorithm + '/' + str_state + '-' + str_goal + '.txt', 'w') as file:
                        file.write(''.join(map(str, solution)))

                control_buttons[5].config(state=tk.NORMAL)
                total_steps = len(solution)
                update_total_steps_count(total_steps)
                exit_btn.config(state=tk.NORMAL)

                i = 0
                str_solution_1 = " " + str(selected_algorithm) + ":\n  "
                str_solution_2 = " " + str(selected_algorithm) + ":\n  "

                for move_to in solution:
                    if step_count > 40 and speed == 0:
                        puzzle = list(goal)
                        step_count = total_steps
                        update_display()
                        update_step_count(step_count)
                        break
                    
                    direction = ""
                    z = puzzle.index(0) - puzzle.index(move_to)
                    if z == ROW:
                        direction = "T"
                    elif z == -ROW:
                        direction = "B"
                    elif z == -1:
                        direction = "R"
                    else:
                        direction = "L"

                    if i <= 12:
                        str_solution_1 += "->" + str(move_to)
                        str_solution_2 += "->" + direction
                        i += 1    
                    else:
                        str_solution_1 += "\n  ->" + str(move_to) 
                        str_solution_2 += "\n  ->" + direction 
                        i = 1

                    move(puzzle, move_to)
                    step_count += 1
                    update_step_count(step_count)
                    game_window.update()
                    time.sleep(speed)

                str_solution_1 += "\n"
                str_solution_2 += "\n"
                add_solution(str_state, str_goal, str_solution_1, str_solution_2)
                add_history(str_state, str_goal)
            else:
                total_steps = 0
                update_total_steps_count(total_steps)
                messagebox.showwarning("Cảnh báo", "Không tìm thấy lời giải!")
        else:
            with open('Data/' + selected_algorithm + '/' + str_state + '-' + str_goal + '.txt', 'r') as file:
                str_solution = file.read()
            solution = [int(move_to) for move_to in str_solution]
        
            if stop_event:
                stop_event.set()
                thread_count.join()

            update_solving_time(solving_time)
            update_infor_lables()
             
            control_buttons[5].config(state=tk.NORMAL)
            total_steps = len(solution)
            update_total_steps_count(total_steps)
            exit_btn.config(state=tk.NORMAL)

            for move_to in solution:
                if step_count > 100 and speed == 0:
                    puzzle = list(goal)
                    step_count = total_steps
                    update_display()
                    update_step_count(step_count)
                    break

                move(puzzle, move_to)
                step_count += 1
                update_step_count(step_count)
                game_window.update()
                time.sleep(speed)
        
        enable_controls()
        control_buttons[5].config(state=tk.DISABLED)
        control_buttons[0]['text'] = "Giải"


    def map_image(puzzle_image):
        global image_mapping
        puzzle_pieces = []
        for i in range(ROW):
            for j in range(COL):
                cropped_image = puzzle_image.crop(
                    (j * SIZE, i * SIZE, (j + 1) * SIZE, (i + 1) * SIZE)
                )
                puzzle_pieces.append(ImageTk.PhotoImage(cropped_image))
        image_mapping = dict(zip(list(range(0, ROW * COL)), puzzle_pieces))

    def btn_upload_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            puzzle_image = Image.open(file_path).resize((COL * SIZE, ROW * SIZE))
            map_image(puzzle_image)
            photo = ImageTk.PhotoImage(Image.open(file_path).resize((215, 215)))
            image_label.config(image=photo)
            image_label.image = photo
            image_label.grid_remove()
            update_display()
        control_buttons[6].config(state=tk.NORMAL)

    def btn_show_image():
        if image_label.winfo_ismapped():
            image_label.grid_remove()
            control_buttons[6]['text'] = "Hiện ảnh gợi ý"
        else:
            image_label.grid()
            control_buttons[6]['text'] = "Ẩn ảnh gợi ý"
        update_display()

    def run_stopwatch():
        global stop_event
        start_time = time.time()
        while not stop_event.is_set():
            update_infor_lables()
            elapsed_time = time.time() - start_time
            time_label.config(text=f"{elapsed_time:.2f}s")
            time.sleep(0.16)

    def btn_reset_click():
        global puzzle
        puzzle = list(saved_state)
        reset_infor_lables()
        update_display()

    def btn_solve_click():
        global depth_limit
        validate_depth()
        depth_limit = int(spinbox.get())
        update_infor_lables()

        global stop_event, thread_count, thread_solve
        if not is_solved(puzzle):
            stop_event = threading.Event()
            thread_solve = threading.Thread(target=run_algorithm)
            thread_solve.daemon = True
            thread_count = threading.Thread(target=run_stopwatch)
            thread_count.daemon = True
            thread_solve.start()
            thread_count.start()
        else:
            messagebox.showinfo("Thông tin", "Puzzle này đã được giải!")

    def btn_speed_click():
        global speed
        speed = 0
        
    def btn_shuffle_click():
        delete_file()
        random_shuffle(puzzle)

        global saved_state
        saved_state = list(puzzle)

        update_solving_time(solving_time)
        reset_infor_lables()

        game_window.update()

    def btn_save_click():
        global saved_state
        saved_state = list(puzzle)

        reset_infor_lables()
        game_window.update()


    game_window = tk.Tk()
    game_window.bind("<KeyPress>", on_key_press)
    game_window.title("N-Puzzle")

    puzzle_image = create_square_image().resize((COL * SIZE, ROW * SIZE))
    map_image(puzzle_image)

    frame = tk.Frame(game_window)
    frame.pack()

    frame1 = tk.Frame(frame)
    frame1.grid(row=0, column=0, columnspan=game_window.winfo_screenwidth())
    game_label = tk.Label(frame1, text="", font=("Tahoma", 15, "bold"), fg="red")
    game_label.grid(row=0, column=0, columnspan=game_window.winfo_screenwidth(), pady=0)

    header_labels = ["Tổng số bước", "Bước hiện tại", "Thời gian giải", "Tổng số đỉnh đã duyệt"]
    for col, label_text in enumerate(header_labels):
        header_label = tk.Label(frame1, text=label_text, font=("Helvetica", 20, "bold"))
        header_label.grid(row=1, column=col, padx=20, pady=5)

    # Infor Lables
    total_steps_label = tk.Label(frame1, text="0", font=("Helvetica", 20))
    total_steps_label.grid(row=2, column=0, pady=0)

    step_label = tk.Label(frame1, text="0", font=("Helvetica", 20))
    step_label.grid(row=2, column=1, pady=0)

    time_label = tk.Label(frame1, text="0.00s", font=("Helvetica", 20))
    time_label.grid(row=2, column=2, pady=0)

    total_nodes_label = tk.Label(frame1, text="0", font=("Helvetica", 20))
    total_nodes_label.grid(row=2, column=3, pady=0)

    depth_limit_header_label = tk.Label(
        frame1, text="Độ sâu tối đa", font=("Helvetica", 20, "bold")
    )
    depth_limit_label = tk.Label(frame1, text="0", font=("Helvetica", 20))

    frame2 = tk.Frame(frame)
    frame2.grid(row=5, column=0, pady=10, columnspan=game_window.winfo_screenwidth())

    optional_label = tk.Label(frame2, font=("Helvetica", 20, "bold"))
    spinbox = tk.Spinbox(
        frame2,
        from_=1,
        to=1000,
        width=12,
        font=("Helvetica", 20),
        command=on_spinbox_change,
    )

    heuristic_combobox = ttk.Combobox(
        frame2,
        values=["Manhattan", "Hamming", "Linear Conflict"],
    )
    heuristic_combobox.configure(width=20, font=("Helvetica", 20))
    heuristic_combobox.set("Manhattan")
    heuristic_combobox.state(["readonly"])
    heuristic_combobox.bind("<<ComboboxSelected>>", on_combobox_change)

    heuristic_rb = tk.StringVar(value="manhattan")
    hamming_rb = tk.Radiobutton(
        frame2,
        text="Hamming",
        variable=heuristic_rb,
        value="hamming",
        font=("Helvetica", 15),
    )
    manhattan_rb = tk.Radiobutton(
        frame2,
        text="Manhattan",
        variable=heuristic_rb,
        value="manhattan",
        font=("Helvetica", 15),
    )
    linear_conflict_rb = tk.Radiobutton(
        frame2,
        text="Linear Conflict",
        variable=heuristic_rb,
        value="linear conflict",
        font=("Helvetica", 15),
    )

    # Control Buttons - Combobox
    algorithm_label = tk.Label(
        frame2, text="Thuật toán:", font=("Helvetica", 20, "bold")
    )
    algorithm_label.grid(row=5, column=1, padx=5, pady=5)

    algorithm_combobox = ttk.Combobox(
        frame2,
        values=["BFS", "DFS", "DLS", "ID", "UCS", "Greedy", "A-Star", "IDA-Star", "Hill Climbing", "Beam Search"],
    )
    algorithm_combobox.configure(width=12, font=("Helvetica", 20))
    algorithm_combobox.set("BFS")
    algorithm_combobox.state(["readonly"])
    algorithm_combobox.bind("<<ComboboxSelected>>", on_combobox_change)
    algorithm_combobox.grid(row=5, column=2, padx=5, pady=5)

    buttons_data = [
        ("Giải", 5, 3, btn_solve_click, "orange", 20, 1),
        ("Tải ảnh", 3, 1, btn_upload_image, "lightblue", 20, 1),
        ("Thay đổi kích thước", 3, 2, btn_change_size, "yellow", 20, 1),
        ("Xáo trộn", 3, 3, btn_shuffle_click, "lime", 20, 1),
        ("Đặt lại", 3, 4, btn_reset_click, "pink", 20, 2),
        ("Tăng tốc", 5, 4, btn_speed_click, "lightgreen", 20, 1),
        ("Hiện ảnh gợi ý", 4, 1, btn_show_image, "lightpink", 20, 1),
        ("Lưu trạng thái", 4, 2, btn_save_click, "lightyellow", 20, 1),
    ]

    control_buttons = []
    for text, row, column, command, bg_color, size, span in buttons_data:
        button = tk.Button(
            frame2,
            text=text,
            width=size,
            height=2,
            font=("Helvetica", 12, "bold"),
            command=command,
            bg=bg_color,
        )
        button.grid(row=row, column=column, padx=5, pady=5, columnspan=span)
        control_buttons.append(button)
    control_buttons[5].config(state=tk.DISABLED)
    control_buttons[6].config(state=tk.DISABLED)

    # Exit Button
    exit_btn = tk.Button(
        frame2,
        text="Thoát",
        width=42,
        height=2,
        bg="#eb1d02",
        font=("Helvetica", 12, "bold"),
        command=btn_exit_click,
        state=tk.NORMAL,
        )
    exit_btn.grid(row=6, column=3, columnspan= 2 ,padx=5, pady=5)

    # History Button
    history_btn = tk.Button(
        frame2, 
        text="Lịch sử", 
        width=20,
        height=2,
        bg="lightgray",
        font=("Helvetica", 12, "bold"),
        command=btn_history_click,
        state=tk.NORMAL,
        )
    history_btn.grid(row=4, column=4, padx=5, pady=5)

    # Solution Button
    solution_btn = tk.Button(
        frame2, 
        text="Hướng dẫn giải", 
        width=20,
        height=2,
        bg="gray",
        font=("Helvetica", 12, "bold"),
        command=btn_solution_click,
        state=tk.NORMAL,
        )
    solution_btn.grid(row=4, column=3, padx=5, pady=5)

    # Puzzle Buttons
    puzzle_frame = tk.Frame(frame)
    puzzle_frame.grid(
        row=1,
        column=0,
        columnspan=game_window.winfo_screenwidth(),
        padx=10,
        pady=10,
    )
    for i in range(ROW):
        row = []
        for j in range(COL):
            button = tk.Button(puzzle_frame, image=image_mapping[i * COL + j])
            button.grid(row=i, column=j, padx=2, pady=2)
            button.config(command=lambda row=i, col=j: on_button_click(row, col))
            row.append(button)
        buttons.append(row)

    update_display()
    update_infor_lables()

    # Image
    image_label = tk.Label(frame)
    image_label.grid(row=1, column=0, rowspan=1)

    window_width = 1050
    window_height = 765

    center_window(game_window, window_width, window_height)

    game_window.mainloop()


def btn_solution_click():
    global solution_window
    def open_file_1():
        file_path = 'Solution/solution_1.txt'
        # file_path = os.path.join(os.path.dirname(__file__), "solution_1.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text.delete(1.0, tk.END)
                text.insert(tk.END, content)
        else:
            text.delete(1.0, tk.END)
            text.insert(tk.END, "Không tìm thấy dữ liệu hướng dẫn giải!\n\nCần thực hiện giải bằng thuật toán trước!")

    def open_file_2():
        file_path = 'Solution/solution_2.txt'
        # file_path = os.path.join(os.path.dirname(__file__), "solution_2.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text.delete(1.0, tk.END)
                text.insert(tk.END, content)
        else:
            text.delete(1.0, tk.END)
            text.insert(tk.END, "Không tìm thấy dữ liệu hướng dẫn giải!\n\nCần thực hiện giải bằng thuật toán trước!")

    solution_window = tk.Tk()
    solution_window.title("Hướng dẫn giải")

    update_button_1 = tk.Button(solution_window, text="Cập nhật dữ liệu hướng dẫn giải dạng số", command=open_file_1)
    update_button_1.pack(pady=5)

    update_button_2 = tk.Button(solution_window, text="Cập nhật dữ liệu hướng dẫn giải dạng hướng", command=open_file_2)
    update_button_2.pack(pady=5)

    text = tk.Text(solution_window, height=15, width=42)
    text.pack(pady=5)

    window_width = 370
    window_height = 335

    center_window(solution_window, window_width, window_height)


def btn_history_click():
    global history_window
    def open_file():
        # file_path = os.path.join(os.path.dirname(__file__), "history.txt")
        file_path = 'History/history.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text.delete(1.0, tk.END)
                text.insert(tk.END, content)
        else:
            text.delete(1.0, tk.END)
            text.insert(tk.END, "Không tìm thấy dữ liệu lịch sử giải!\n\nCần thực hiện giải bằng thuật toán trước!")

    history_window = tk.Tk()
    history_window.title("Lịch sử")

    update_button = tk.Button(history_window, text="Cập nhật dữ liệu", command=open_file)
    update_button.pack(pady=10)

    text = tk.Text(history_window, height=15, width=90)
    text.pack()

    window_width = 775
    window_height = 315

    center_window(history_window, window_width, window_height)
    open_file()


def btn_change_size():
    game_window.destroy()
    delete_file()
    os.system('python {}'.format(' '.join(sys.argv)))


def btn_exit_click():
    delete_file()
    sys.exit()


def delete_file():
    if os.path.exists('History/history.txt'):
        os.remove('History/history.txt')
    if os.path.exists('Solution/solution_1.txt'):
        os.remove('Solution/solution_1.txt')
    if os.path.exists('Solution/solution_2.txt'):
        os.remove('Solution/solution_2.txt')


def main_menu():
    global row_combobox, col_combobox, menu_window

    menu_window = tk.Tk()
    menu_window.title("N-Puzzle")

    menu_frame = tk.Frame(menu_window)
    menu_frame.pack()

    tk.Label(menu_frame, text="Nhóm 11", 
             fg="black", font=("Tahoma", 25, "bold")).grid(
                 row=0, column=0, padx=25, pady=15
    )
    tk.Label(menu_frame, text="Nguyễn Khánh Quy - 21110282\nNguyễn Hồng Thông Điệp - 21110166\nVõ Chí Khương - 21110221", 
             fg="black", font=("Tahoma", 16)).grid(
                 row=1, column=0, padx=25, pady=0
    )
    tk.Label(menu_frame, text="Thiết kế và xây dựng game N-Puzzle\nsử dụng thuật toán AI", 
             fg="black", font=("Tahoma", 23, "italic")).grid(
                 row=2, column=0, padx=25, pady=15
    )

    frame = tk.Frame(menu_frame)
    frame.grid(row=3, column=0, padx=25, pady=25)

    tk.Label(frame, text="Kích thước hàng:", font=("Helvetica", 20, "bold")).grid(
        row=0, column=0, padx=5, pady=5, sticky='w'
    )
    row_combobox = ttk.Combobox(
        frame, values=[2, 3, 4, 5], width=3, font=("Helvetica", 20)
    )
    row_combobox.grid(row=0, column=1, padx=5, pady=5)
    row_combobox.set(3)
    row_combobox.state(["readonly"])

    tk.Label(frame, text="Kích thước cột:", font=("Helvetica", 20, "bold")).grid(
        row=1, column=0, padx=5, pady=5, sticky='w'
    )
    col_combobox = ttk.Combobox(
        frame, values=[2, 3, 4, 5], width=3, font=("Helvetica", 20)
    )
    col_combobox.grid(row=1, column=1, padx=5, pady=5)
    col_combobox.set(3)
    col_combobox.state(["readonly"])

    tk.Button(
        menu_frame,
        text="Tạo Puzzle",
        width=15,
        height=1,
        bg="#7af218",
        font=("Helvetica", 20, "bold"),
        command=prepare_puzzle,
    ).grid(row=4, column=0, padx=0, pady=0)

    tk.Button(
        menu_frame,
        text="Thoát",
        width=15,
        height=1,
        bg="#eb1d02",
        font=("Helvetica", 20, "bold"),
        command=btn_exit_click,
    ).grid(row=5, column=0, padx=0, pady=20)

    window_width = 600
    window_height = 600

    center_window(menu_window, window_width, window_height)
    
    menu_window.mainloop()


main_menu()
