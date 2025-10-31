import os
import subprocess

print (os.getcwd())

for SF in range(15, 31, 5):
  for qid in range(12 if SF == 15 else 1, 22+1):
    with open(f"./experiments/results/Q{qid}SF{SF}.txt", "w") as f:
      subprocess.run(["conda", "run", "-n", "tqp", "python", "main.py", "--SF", f"{SF}", "--q", f"{qid}"], stdout=f, stderr=f)
    print (f"QID = {qid}, SF = {SF} Finish.")