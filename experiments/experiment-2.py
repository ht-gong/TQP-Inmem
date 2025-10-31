import os
import subprocess

print (os.getcwd())

qids = range(1, 22)

print (f"qids = {qids}, Enter to confirm.")
input()
# SF100 Q21 = 26, other = 27

print ("Start.")
info = ""
for round in range(1):
  # for SF in [500]:
  for SF in [1]:
    for qid in qids:
      dat = 'May27'
      path = f"./experiments/{dat}/results/"
      if not os.path.exists(path):
        os.makedirs(path)
      
      path = f"./experiments/{dat}/figures/"
      if not os.path.exists(path):
        os.makedirs(path)

      log_path = f"./experiments/{dat}/results/q{qid}sf{SF}.log"
      tee_path = f"./experiments/{dat}/results/q{qid}sf{SF}.txt"
      
      cmd = (
          f"numactl --cpunodebind=0 --membind=0 "
          f"/home1/zhaoyah/miniconda/envs/env/bin/python "
          f"/work1/talati/zhaoyah/TQP-Vortex/main.py "
          f"--SF {SF} --q {qid} --h {30 if SF == 500 else 27} --log {log_path} | tee {tee_path}"
      )
      subprocess.run(cmd, shell=True, executable="/bin/bash")

      cmd = (
        f"python ./experiments/parse_log.py {log_path} ./experiments/{dat}/figures/ q{qid}sf{SF}"
      )
      subprocess.run(cmd, shell=True, executable="/bin/bash")

      with open(tee_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip() if lines else ""
        info += f"Q{qid}" + last_line + '\n'

print (info)