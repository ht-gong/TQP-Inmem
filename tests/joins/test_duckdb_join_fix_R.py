import duckdb
import time
import json

join_type = 'INNER'

for num_threads in [64]:
  for R in [1]:
    # for N in [10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:
    for N in [5 * 10**8]:
      con = duckdb.connect()
      con.execute(f"PRAGMA threads={num_threads}")

      con.execute(f"""
          CREATE TABLE A AS
          SELECT
              i AS id,             -- join key
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
          SELECT COUNT(*) FROM A {join_type} JOIN B USING(id)
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
