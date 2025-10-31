from multiprocessing import set_start_method, Process, Queue
from threading import Thread
import time
from main import exec


SF = 1
h = 30 if SF == 500 else 27
num_of_queries = 22
num_of_workers = 1

def worker(rank, task_queue, result_queue):
  global maps
  while True:
    task = task_queue.get()
    if task == -1:
      print(f"Worker {rank} done.")
      break
    print(f"Worker {rank} got task: {task}")
    result_queue.put((task, rank))
    exec(rank, task, SF, True, 1 << h)
    print(f"Worker {rank} task {task} finished")
  
  # for task in range(rank, 23, num_of_workers):
    # print (f"T = {task}")
    # if task:
    #   try:
    #       # exec(rank, task, SF, True, 1 << h)
    #       maps[task] = rank
    #   except Exception as e:
    #       print(f"Task {task} on rank {rank} failed: {e}", flush=True)

      # exec(rank, task, SF, True, 1 << h)

if __name__ == "__main__":
  set_start_method("spawn", force=True)
  
  task_queue = Queue()
  result_queue = Queue()
    
  processes = []
  for i in range(num_of_workers):
    p = Process(target=worker, args=(i, task_queue, result_queue))
    processes.append(p)

  for p in processes:
    p.start()

  time.sleep(10)
  print ("BEGIN -------------------------------------")
  time_l = time.time()

  for i in range(num_of_queries):
    task_queue.put(i+1)
  for i in range(num_of_workers):
    task_queue.put(-1)
  for p in processes:
    p.join()
  time_r = time.time()
  # time_r = time.time()

  # time_l = time.time()
  # threads = [Thread(target=worker, args=(_,)) for _ in range(num_of_workers)]
  # for t in threads:
  #   t.start()
  # for t in threads:
  #   t.join()
  while not result_queue.empty():
    task, worker = result_queue.get()
    print (f"task {task} finished by {worker}")
  print (f"Time Total = {time_r - time_l}", flush=True)