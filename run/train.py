from utils.parallel import get_device
from models.save import save_statedict
from models.load import load_model,update_statedict
from utils.parallel import get_device
from data.load import get_data
import time
import torch
import numpy as np
import sys

from utils.slurm import flushed_print


class Timer:
    def __init__(self,):
        self.times = {}
    def start(self,label):
        # flushed_print(label,' got initiated')
        if label not in self.times:
            self.times[label] = []
        self.times[label].append(time.time())
    def end(self,label):
        # flushed_print(label,' ended')
        assert label in self.times
        t1 = self.times[label][-1]
        self.times[label][-1] = time.time() - t1
    def __repr__(self) -> str:
        keys = [f"\t{lbl} : {np.mean(vals[-5:])}" for lbl, vals in self.times.items()]
        return "\n".join(keys)
    def reset(self,):
        self.times = {}
def preprocess(infields,outfields,mask,device,linsupres = False,return_true_force = False):
    if linsupres:
        nchans = outfields.shape[1]//2
        lsrpres = outfields[:,nchans:]
        lsrp = outfields[:,:nchans]
        mask = mask[:,nchans:]
        mask = mask.to(device)
        infields = infields.to(device)
        lsrpres =lsrpres.to(device)
        if return_true_force:
            lsrp = lsrp.to(device)
        return infields,(lsrp,lsrpres),mask
    else:
        forcing = outfields
        mask = mask.to(device)
        infields = infields.to(device)
        forcing =forcing.to(device)
        return infields,forcing,mask
def prob_outputs(outputs,outfields,mask):
    if isinstance(outputs,tuple):
        mean,_ = outputs
    out = mean.detach().to("cpu")
    m = mask.detach().to("cpu")
    return {'out-absval': torch.mean(torch.abs(out[m>0.5])).item(),
            'true-absval': torch.mean(torch.abs(outfields[m>0.5])).item()}


def cnn_train(args):
    modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs=load_model(args)
    flushed_print('torch.cuda.is_available():\t',torch.cuda.is_available())
    flushed_print('runargs:\t',runargs)
    training_generator,val_generator=get_data(args,half_spread = net.spread,torch_flag = True,data_loaders = True,groups = ('train','validation'))
    device=get_device()
    net.to(device)
    print(f"using device: {device}")
    flushed_print("epochs started")
    timer = Timer()

    for epoch in range(runargs.epoch,runargs.maxepoch):
        logs['train-loss'].append([])
        tt=0
        net.train()
        timer.start('data')
        for infields,outfields,mask in training_generator:
            if not torch.any(mask>0):
                continue
            infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
            timer.end('data')
            timer.start('model')
            outputs = net.forward(infields)
            loss = criterion(outputs, outfields, mask)
            logs['train-loss'][-1].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            timer.end('model')
            if runargs.sanity:
                flushed_print(prob_outputs(outputs,outfields,mask))


            tt+=1
            if runargs.disp > 0 and tt%runargs.disp==0:
                flushed_print('\t\t\t train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),\
                        '\t ±',\
                        str(np.std(np.array(logs['train-loss'][-1]))))
                flushed_print(timer)
            timer.start('data')
            # if len(logs['train-loss'][-1]) == 4:
            #     break

        timer.reset()
        # if runargs.sanity:
        #     continue
        with torch.set_grad_enabled(False):
            net.eval()
            val_loss=0.
            num_val=0
            for infields,outfields,mask in val_generator:
                if not torch.any(mask>0):
                    continue
                infields,outfields,mask = infields.to(device),outfields.to(device),mask.to(device)
                outputs = net.forward(infields)
                loss = criterion(outputs, outfields, mask)
                val_loss+=loss.item()
                num_val+=1
        logs['val-loss'].append(val_loss/num_val)
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step(logs['val-loss'][-1])



        if len(logs['epoch'])>0:
            logs['epoch'].append(epoch)
        else:
            logs['epoch'].append(0)

        flushed_print('#epoch ',str(logs['epoch'][-1]),' ',\
                    ' val-loss: ',str(logs['val-loss'][-1]),\
                    ' train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),\
                    ' lr: ',str(logs['lr'][-1]))

        state_dict = update_statedict(state_dict,net,optimizer,scheduler,last_model = True)
        if np.amin(logs['val-loss']) == logs['val-loss'][-1]:
            state_dict = update_statedict(state_dict,net,optimizer,scheduler,last_model = False)
        save_statedict(modelid,state_dict,logs)
        if logs['lr'][-1]<1e-7:
            break


def main():
    args = sys.argv[1:]
    cnn_train(args)

if __name__=='__main__':
    main()
