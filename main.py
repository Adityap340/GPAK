import ctypes
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the shared library compiled from CUDA code
pagerank_lib = ctypes.CDLL('C:\\Users\\Aditya\\Desktop\\Katkar\\pagerank.dll')

# Function to wrap CUDA kernel execution via shared library
def run_pagerank_cuda(graph, num_nodes, damping_factor, max_iterations):
    graph_np = np.array(graph, dtype=np.float32)
    pagerank_scores = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)

    graph_ptr = graph_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pagerank_ptr = pagerank_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Run CUDA-based PageRank
    pagerank_lib.run_pagerank(graph_ptr, ctypes.c_int(num_nodes), ctypes.c_float(damping_factor),
                              ctypes.c_int(max_iterations), pagerank_ptr)

    return pagerank_scores.tolist()

# Python-based PageRank calculation
def run_pagerank_python(graph, num_nodes, damping_factor, max_iterations):
    pagerank_scores = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)

    for _ in range(max_iterations):
        temp_scores = np.zeros(num_nodes, dtype=np.float32)
        for i in range(num_nodes):
            temp_sum = 0.0
            for j in range(num_nodes):
                if graph[j][i] != 0:
                    temp_sum += pagerank_scores[j] / np.sum(graph[j])
            temp_scores[i] = (1 - damping_factor) + damping_factor * temp_sum
        pagerank_scores = temp_scores

    return pagerank_scores.tolist()

# Initialize the matrix and setup GUI
root = tk.Tk()
root.title("CUDA vs Python PageRank Execution Time")
root.state('zoomed')

# Define parameters
damping_factor = 0.85
max_iterations = 100
graph_sizes = [20, 40, 100]  # Graph sizes for testing

# Configure UI elements
bg_color = "#1e272e"
text_color = "#485460"
frame_color = "#d2dae2"
highlight_color = "#00a8ff"

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background=bg_color, foreground=text_color, font=("Helvetica", 14, "bold"))
style.configure("TButton", background=highlight_color, foreground=bg_color, font=("Helvetica", 12, "bold"), padding=5)
style.configure("TFrame", background=bg_color)

input_frame = ttk.Frame(root, padding="20", style="TFrame")
input_frame.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
output_frame = ttk.Frame(root, padding="20", style="TFrame")
output_frame.grid(row=1, column=0, padx=30, pady=30, sticky="nsew")

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

output_log = tk.Text(output_frame, height=20, wrap="word", bg=frame_color, fg=text_color, font=("Helvetica", 12))
output_log.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Function to display PageRank scores in a popup window
def display_pagerank_scores(cuda_scores, python_scores, size):
    scores_window = tk.Toplevel(root)
    scores_window.title(f"PageRank Scores (CUDA vs Python) for {size}x{size}")

    # Adding a text box for the scores
    scores_text = tk.Text(scores_window, wrap="word", font=("Helvetica", 12), bg=frame_color, fg=text_color)
    scores_text.pack(fill="both", expand=True)

    # Insert CUDA and Python scores
    scores_text.insert("end", "CUDA PageRank Scores:\n")
    scores_text.insert("end", str(cuda_scores) + "\n")  # Convert to string for display
    scores_text.insert("end", "\n" + "-"*50 + "\n")
    scores_text.insert("end", "Python PageRank Scores:\n")
    scores_text.insert("end", str(python_scores))  # Convert to string for display

# Function to run and compare PageRank on CUDA and Python
def run_pagerank():
    cuda_times = []
    python_times = []

    for size in graph_sizes:
        graph = np.random.rand(size, size).astype(np.float32)

        # CUDA execution
        start_cuda = time.time()
        cuda_scores = run_pagerank_cuda(graph, size, damping_factor, max_iterations)
        cuda_time = time.time() - start_cuda
        cuda_times.append(cuda_time)
        output_log.insert("end", f"{size}x{size} CUDA Execution Time: {cuda_time:.4f} seconds\n")

        # Python execution
        start_python = time.time()
        python_scores = run_pagerank_python(graph, size, damping_factor, max_iterations)
        python_time = time.time() - start_python
        python_times.append(python_time)
        output_log.insert("end", f"{size}x{size} Python Execution Time: {python_time:.4f} seconds\n")

        # Display scores in pop-up
        display_pagerank_scores(cuda_scores, python_scores, size)

    output_log.insert("end", "-" * 60 + "\n")
    output_log.see("end")

    # Display timing differences in side-by-side graph
    visualize_execution_times(cuda_times, python_times)

# Visualization function for CUDA and Python execution time comparison
def visualize_execution_times(cuda_times, python_times):
    for widget in output_frame.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    indices = np.arange(len(graph_sizes))
    
    ax.bar(indices - bar_width / 2, cuda_times, bar_width, label="CUDA", color=highlight_color)
    ax.bar(indices + bar_width / 2, python_times, bar_width, label="Python", color='orange')
    
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{size}x{size}" for size in graph_sizes])
    ax.set_xlabel("Graph Size")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("CUDA vs Python PageRank Execution Time by Graph Size")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=output_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

run_button = ttk.Button(input_frame, text="Run PageRank for All Sizes", command=run_pagerank)
run_button.grid(row=4, column=0, pady=20)

input_frame.grid_columnconfigure(0, weight=1)
output_frame.grid_rowconfigure(0, weight=1)
output_frame.grid_columnconfigure(0, weight=1)

root.mainloop()
