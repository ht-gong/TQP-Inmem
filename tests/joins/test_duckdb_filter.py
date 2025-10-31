import duckdb
import time
import json

# 配置参数
N = 1_000_000_000_0  # 总行数
N = 3000_000_000
selectivities = [0.2, 0.4, 0.6, 0.8, 1]  # 控制过滤率
threads_list = [8, 32]

con = duckdb.connect(f'rows{N}.duckdb')
con.execute(f"PRAGMA threads={64}")
con.execute(f"""
    CREATE TABLE IF NOT EXISTS A AS
    SELECT
        i AS id,
        RANDOM() AS score
    FROM range(0, {N}) tbl(i)
""")
con.close()

for num_threads in threads_list:
    for sel in selectivities:
        con = duckdb.connect(f"rows{N}.duckdb")
        con.execute(f"PRAGMA threads={num_threads}")
        threshold = 1.0 - sel

        result = con.execute(f"SELECT COUNT(*) FROM A WHERE id  BETWEEN 222 AND 100000").fetchall()

        start = time.time()
        result = con.execute(f"EXPLAIN ANALYZE SELECT * FROM A WHERE score > {threshold}").fetchall()
        end = time.time()

        print(f"[threads={num_threads} | selectivity={sel}] : {end - start}")
        print ("result = ", result)
        

        benchmark_result = {
            "time": end - start
        }

        with open(f"./tests/results/duckdb_filter_{num_threads}threads_{int(sel*100)}pct_{N}rows.json", 'w') as f:
            json.dump(benchmark_result, f, indent=2)
        
        con.close()
