import torch
import constants
from conversion import normalize, rearrange_tensors

def tqp_sortjoin(gpu_enable, tensorsa, tensorsb, leftcol, rightcol):
    
    device_name = 'cuda' if gpu_enable else 'cpu'

    left, right = normalize(tensorsa[leftcol].tensor), normalize(tensorsb[rightcol].tensor)
    assert left.dim() == 1 and right.dim() == 1

    left, leftidx = torch.sort(left)
    right, rightidx = torch.sort(right)
    

    minLength = max(left[-1], right[-1]) + 1

    leftHist = torch.bincount(left, minlength=minLength)
    rightHist = torch.bincount(right, minlength=minLength)

    histMul = torch.mul(leftHist, rightHist)

    cumLeft = torch.cumsum(leftHist, dim=0)
    cumRight = torch.cumsum(rightHist, dim=0)
    cumMul = torch.cumsum(histMul, dim=0)
    outSize = cumMul[-1]

    offset = torch.arange(outSize, device=device_name)

    outBucket = torch.bucketize(offset, cumMul, right=True)

    offset.sub_(cumMul[outBucket] - histMul[outBucket])

    leftoutidx = leftidx[cumLeft[outBucket] - leftHist[outBucket] +
                 torch.div(offset, rightHist[outBucket], rounding_mode = "floor")]
    
    rightoutidx = rightidx[cumRight[outBucket] - rightHist[outBucket] + 
        torch.remainder(offset, rightHist[outBucket])]
            
    rearrange_tensors(tensorsa, leftoutidx)
    rearrange_tensors(tensorsb, rightoutidx)

    return list(tensorsa.values())[0].tensor.shape[0]
