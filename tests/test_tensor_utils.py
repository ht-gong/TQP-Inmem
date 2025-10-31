import utility.tensor_utils
from variable import Variable
import torch

import torch

def test():
    # deterministic randoms
    g = torch.Generator(device="cpu").manual_seed(1234)

    numel = 5
    repeat = 2

    # Assuming your Variable class takes (tensor, name)
    left  = [Variable(torch.sort(torch.randint(1, 100, (numel,), dtype=torch.int64, generator=g))[0], '') for _ in range(repeat)]
    right = [Variable(torch.sort(torch.randint(1, 100, (numel,), dtype=torch.int64, generator=g))[0], '') for _ in range(repeat)]
    for l in left:
        print(l.tensor)
    for r in right:
        print(r.tensor)

    l_res, r_res = utility.tensor_utils.find_boundaries_for_join(left, right, numel)

    print(l_res, r_res)
    return
    # l_res / r_res are cumulative cut positions per side; verify they stay in range
    total = 0
    for i in range(len(l_res)):
        assert l_res[i] <= numel, f"Left boundary {l_res[i]} exceeds numel {numel}"
        assert r_res[i] <= numel, f"Right boundary {r_res[i]} exceeds numel {numel}"
        total += l_res[i] if i == 0 else (l_res[i] - l_res[i - 1])
        total += r_res[i] if i == 0 else (r_res[i] - r_res[i - 1])

    assert total == numel * repeat, f"Total accounted {total} vs expected {numel * repeat}"

def main():
    test()
