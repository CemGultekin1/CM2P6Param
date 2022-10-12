import numpy as np
import torch
import time
def get_device():
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark=True
    return device

def random_wait(secs=2,parallel=True):
    tt=np.random.rand()*5
    time.sleep(tt)