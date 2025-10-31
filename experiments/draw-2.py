import re
import matplotlib.pyplot as plt
# import seaborn as sns

# Set up a more polished theme with seaborn
# sns.set_theme(style="ticks", palette="muted", font_scale=1.3)

# def plot_queries(start_query, end_query):
#     sf = list(range(5, 31, 5))

#     # Create a larger figure and arrange subplots
#     fig, axs = plt.subplots(4, 2, figsize=(16, 20))
#     axs = axs.flatten()

#     # Define color palette for better visual appeal
#     colors = sns.color_palette("bright", 2)

#     # Iterate through the queries
#     for idx, query_id in enumerate(range(start_query, end_query + 1)):
#         transfer_times = []
#         exec_times = []

#         for i in sf:
#             file_path = f"./experiments/results/Q{query_id}SF{i}.txt"
#             try:
#                 with open(file_path, "r") as file:
#                     lines = [line.strip() for line in file if line.strip()]
#                     if not lines:
#                         raise ValueError(f"{file_path} is empty.")
                    
#                     last_line = lines[-1]
#                     match = re.search(r"\[REPORT\]:\s*([\d.]+),\s*([\d.]+)", last_line)
#                     if match:
#                         transfer_time = float(match.group(1))
#                         exec_time = float(match.group(2)) - transfer_time
#                         transfer_times.append(transfer_time)
#                         exec_times.append(exec_time)
#                     else:
#                         raise ValueError(f"Invalid format in {file_path}")
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#                 transfer_times.append(None)
#                 exec_times.append(None)

#         # Plot with distinct colors and enhanced markers
#         ax = axs[idx]
#         ax.plot(sf, transfer_times, label="Transfer Time", marker='o', color=colors[0], markersize=8, linestyle='-', linewidth=2)
#         ax.plot(sf, exec_times, label="Execution Time", marker='s', color=colors[1], markersize=8, linestyle='--', linewidth=2)
        
#         ax.set_xlabel("Scale Factor", fontsize=14)
#         ax.set_ylabel("Time (s)", fontsize=14)
#         ax.set_title(f"Query {query_id}", fontsize=16, fontweight='bold')

#         # Add grid with custom styling
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#         ax.set_facecolor("#f7f7f7")

#         # Adding Legends with better styling
#         ax.legend(loc='upper left', fontsize=12, fancybox=True, framealpha=0.5)

#         # Customize ticks for better visibility
#         ax.tick_params(axis='both', which='major', labelsize=12)
#         ax.tick_params(axis='both', which='minor', labelsize=10)

#     # Remove unused subplots
#     for j in range(idx + 1, 8):
#         fig.delaxes(axs[j])

#     # Improve layout to avoid overlap
#     plt.tight_layout()
#     plt.savefig(f"./experiments/figures/queries_{start_query}_to_{end_query}.png", dpi=300)
#     plt.close()
#     print(f"Plot saved as queries_{start_query}_to_{end_query}.png")

# Generate for each set of 8 queries
# for i in range(1, 23, 8):
#     plot_queries(i, min(i + 7, 22))


i = 100
for query_id in range(1, 22+1):
    file_path = f"./experiments/results/Q{query_id}SF{i}.txt"
    try:
        with open(file_path, "r") as file:
            lines = [line.strip() for line in file if line.strip()]
            if not lines:
                raise ValueError(f"{file_path} is empty.")
            
            last_line = lines[-1]
            print (f"Q{query_id}", last_line)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")