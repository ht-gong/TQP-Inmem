import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("optimized_time.csv")

# 画图
plt.figure(figsize=(8, 5))
plt.plot(df["width"], df["old"], marker="o", label="Old")
plt.plot(df["width"], df["new"], marker="s", label="New")

plt.xlabel("Width")
plt.ylabel("Time (s)")
plt.title("Optimized Time Comparison")
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig("optimized_time.png", dpi=300)
plt.savefig("optimized_time.pdf")  # 可选，保存矢量图
