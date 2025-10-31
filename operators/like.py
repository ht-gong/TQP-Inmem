import torch

def like_contains(a, p1):
  n1 = p1.shape[0]
  n, m = a.shape

  appear = torch.zeros(n, dtype=torch.bool, device=a.device)
  for pos in range(m - n1 + 1):
    appear.logical_or_((a[:, pos:pos+n1] == p1).all(dim=-1))

  return appear

def like_mask(a, p1, p2):
  assert a.dim() == 2 and p1.dim() == 1 and p2.dim() == 1
  n1, n2 = p1.shape[0], p2.shape[0]
  n, m = a.shape
  appear = torch.zeros(n, dtype=torch.bool, device=a.device)
  result = torch.zeros(n, dtype=torch.bool, device=a.device)
  for pos in range(m - n1 - n2 + 1):
    appear.logical_or_((a[:, pos:pos+n1] == p1).all(dim=-1))
    result.logical_or_(torch.logical_and(appear, (a[:, pos+n1:pos+n1+n2] == p2).all(dim=-1)))

  return result