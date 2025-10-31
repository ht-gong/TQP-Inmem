import matplotlib.pyplot as plt
import numpy as np

# Replace these lists with your actual runtime values (in seconds)
labels = ['Q1', 'Q2', 'Q6', 'Q9', 'Q14', 'Q18']
# runtime_tqp_vortex     = [1.761, 2.707, 0.411, 5.658, 0.573, 5.678]  # TQP with Vortex
runtime_tqp_vortex     = [0.295, 1.738, 0.092, 3.306, 0.262, 1.435]  # TQP with Vortex
runtime_tqp_no_vortex  = [0.567, 1.035, 0.154, 1.686, 0.218, 1.286]  # TQP without Vortex
runtime_duckdb         = [0.088, 0.132, 0.140, 0.019, 0.068, 0.534]  # DuckDB

x = np.arange(len(labels))
width = 0.2  # width of the bars

plt.figure()
plt.bar(x - width, runtime_tqp_vortex,    width, label='TQP with Vortex')
plt.bar(x,         runtime_tqp_no_vortex, width, label='TQP without Vortex')
plt.bar(x + width, runtime_duckdb,        width, label='DuckDB')

plt.xticks(x, labels)
plt.xlabel('Query')
plt.ylabel('Runtime (s)')
plt.title('Runtime Comparison Across Queries, SF10')
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('runtime_comparison_barplot_SF10.png')
plt.show()
