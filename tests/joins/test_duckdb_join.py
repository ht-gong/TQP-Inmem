import duckdb
import time
import json

# 配置参数
N = 1_0000_0000
join_type = 'INNER'   # JOIN 类型：INNER, LEFT, RIGHT, etc.

for num_threads in [1, 8, 32]:
  for R in [1, 5, 10, 20]:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={num_threads}")
    # MOD(i, {N // R}) AS id,             -- join key
    con.execute(f"""
        CREATE TABLE A AS
        SELECT
            i AS id,                -- join key
            RANDOM() AS score                
        FROM range(0, {N}) tbl(i)
    """)


    con.execute(f"""
        CREATE TABLE B AS
        SELECT
            i AS id,             -- join key
            RANDOM() AS score
        FROM range(0, {N}) tbl(i)
    """)

    result_rows = con.execute(f"""
        SELECT count(*) FROM A {join_type} JOIN B USING(id)
    """).fetchone()
    print(f"Result rows   : {result_rows[0]}")

    start = time.time()
    result = con.execute(f"""
        EXPLAIN ANALYZE SELECT * FROM A {join_type} JOIN B USING(id)
    """).fetchall()
    end = time.time()

    benchmark_result = {
      "time": end - start
    }

    with open(f"./tests/results/duckdb_{R}repeats_{num_threads}threads_{N}rows.json", 'w') as f:
      json.dump(benchmark_result, f, indent=2)
    
    con.close()
