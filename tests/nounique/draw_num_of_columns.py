import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("experiment_results.csv")

plt.figure(figsize=(8,5))
plt.plot(data["width"], data["percent"], marker='o', linestyle='-')  # 折线图
# plt.bar(data["width"], data["percent"])  # 如果想用柱状图

plt.xlabel("Key Width")
plt.ylabel("Percent (%)")
plt.title("Unique Operator Overhead in Aggregation")
plt.grid(True)

name = "experiment_result.png"
plt.savefig(name, dpi=300) 

print (f"saved to {name}")


data = pd.read_csv("experiment_runtime.csv")

plt.figure(figsize=(8,5))
plt.plot(data["width"], data["time"], marker='o', linestyle='-')  # 折线图

plt.xlabel("Key Width")
plt.ylabel("Time (s)")
plt.title("Total Aggregation Time")
plt.grid(True)

name = "experiment_runtime.png"
plt.savefig(name, dpi=300)

print (f"saved to {name}")