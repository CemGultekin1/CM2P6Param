import functools
import numpy as np
import torch



def ganlossfun(output,target,mask,eps=0):
    '''pout=torch.sum(output*mask,dim=(1,2,3))/torch.sum(mask,dim=(1,2,3))
    return torch.mean(torch.log((pout)*target+(1-target)*(1-pout)+eps))'''
    #pout=torch.sum(output*mask,dim=(1,2,3))/torch.sum(mask,dim=(1,2,3))
    return torch.sum((target-output)**2*mask)/torch.sum(mask)
def negloglikelihood(probs,mask,eps=1e-2):
    numclass=probs.shape[1]
    N=torch.sum(mask)
    #-probs*torch.log10(probs + eps)-(1-probs)*torch.log10(1-probs+eps)
    probs=torch.mean(torch.log10(probs+eps)+torch.log10(1-probs+eps),dim=1,keepdim=True)
    probs=probs[mask>0]
    return torch.sum(probs)/N



def logLoss(output, target,mask,eps=1e-5):
    l2 = torch.log( (target - output)**2 + eps) - np.log(eps)
    l2 = torch.sum(l2,dim=1,keepdim=True)
    l2=l2[mask>0]
    l2 = torch.mean(l2)
    return l2

def mask_decorator(func,):
    @functools.wraps(func)
    def _wrap(output,target,premask):
        mask = premask>0.5
        if isinstance(output,tuple) or isinstance(output,list):
            mean, prec =output
            mean = mean[mask]
            prec = prec[mask]
            output = (mean,prec)
        else:
            output = output[mask]
        target = target[mask]
        loss = func(output,target)
        return loss
    return _wrap

@mask_decorator
def MSE(output, target,):
    if isinstance(output,tuple):
        mean, _ = output
    else:
        mean = output
    l2 =  1 / 2 * (target - mean)**2
    l2=torch.mean(l2)
    return l2

@mask_decorator
def heteroscedasticGaussianLoss(output, target,eps=1e-5):
    mean, precision =output
    precision=precision + eps
    err2=(target - mean)**2
    premeanloss = - 1 / 2 *  torch.log(precision) \
            +  1 / 2 * err2 * precision \
            + 1 / 2 * np.log(1e3)
    loss=torch.mean(premeanloss,)
    return loss
