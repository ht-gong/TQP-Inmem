import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import IO.vortex as vortex

granularity = 20_000_000
n, m = 1001, 203

exchange = vortex.exchange(granularity)

a = torch.ones((n, m), dtype=torch.int64, device='cpu')
a_to = torch.empty((n, m), dtype=torch.int64, device='cuda')

b = torch.full((m, n), -1, dtype=torch.int64, device='cuda')
b_to = torch.empty((m, n), dtype=torch.int64, device='cpu')

exchange.launch([a_to], [a], [b_to], [b])
exchange.sync()

a_to = a_to.to('cpu')
b = b.to('cpu')

print (f"a = {(a == a_to).all()}, b = {(b == b_to).all()}")
