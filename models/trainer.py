from data.load import get_data
from models.save import save_statedict
import torch
from models.load import load_model, update_statedict
from utils.parallel import get_device
from utils.slurm import flushed_print
import time
import numpy as np

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
        keys = [f"\t{lbl} : {np.mean(vals[-30:])}" for lbl, vals in self.times.items()]
        return "\n".join(keys)
    def reset(self,):
        self.times = {}

def prob_outputs(outputs,outfields,mask):
    if isinstance(outputs,tuple):
        mean,_ = outputs
    out = mean.detach().to("cpu")
    m = mask.detach().to("cpu")
    return {'out-absval': torch.mean(torch.abs(out[m>0.5])).item(),
            'true-absval': torch.mean(torch.abs(outfields[m>0.5])).item()}

class BaseTrainer:
    def __init__(self,args):
        self.modelid,_,self.net,self.criterion,self.optimizer,self.scheduler,self.logs,self.runargs=load_model(args)
        flushed_print('torch.cuda.is_available():\t',torch.cuda.is_available())
        flushed_print('runargs:\t',self.runargs)
        self.training_generator,self.val_generator = get_data(args,half_spread = self.net.spread,torch_flag = True,data_loaders = True,groups = ('train','validation'))
        self.device=get_device()
        self.net.to(self.device)
        print(f"using device: {self.device}")
        flushed_print("epochs started")
        self.timer = Timer()
    def dataset_iterator(self,mode :str = 'train'):
        self.timer.start('data')
        tt = 0
        if mode == 'train':
            gen = self.training_generator
        else:
            assert mode == 'val'
            gen = self.val_generator
        for infields,outfields,mask in gen:
            if not torch.any(mask>0):
                continue
            self.timer.end('data')
            yield infields,outfields,mask,tt
            tt+=1
            self.timer.start('data')
        self.timer.reset()

    def plot_train_inner_loop(self,):
        infields = infields[0].numpy()
        outfields = outfields[0].numpy()
        mask = mask[0].numpy()
        import matplotlib.pyplot as plt
        def plot_method(field,name):
            
            nchan = field.shape[0]
            fig,axs = plt.subplots(nchan,1,figsize = (10,10*nchan))
            for i in range(nchan):
                print(name,i,np.mean(np.abs(field[i])))
                ff = field[i]#.numpy()
                ff = ff[::-1]
                axs[i].imshow(ff)
            fig.savefig(name)
            plt.close()
        plot_method(infields,'infields.png')
        plot_method(outfields,'outfields.png')
        plot_method(mask,'mask.png')

    def intervention(self,infields,outfields,mask):
        with torch.set_grad_enabled(False):
            self.net.eval()
            outputs = self.net.forward(self.infields)
        mean,_ = outputs
        yhat = mean.numpy()[0]
        y = outfields.numpy()[0]
        m = mask.numpy()[0] < 0.5
        yhat[m] = np.nan
        y[m] = np.nan
        
        nchan = yhat.shape[0]
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(nchan,2,figsize = (2*5,nchan*6))
        for chani in range(nchan):
            ax = axs[chani,0]
            ax.imshow(y[chani,::-1])
            ax = axs[chani,1]
            ax.imshow(yhat[chani,::-1])
        fig.savefig('train_intervention.png')

    def train_inner_loop(self,infields,outfields,mask):
        self.timer.start('model')
        outputs = self.net.forward(infields)
        outfields,mask = outfields.to(self.device),mask.to(self.device)
        loss = self.criterion(outputs, outfields, mask)
        self.logs['train-loss'][-1].append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.timer.end('model')
        return outputs
    def disp(self,outputs,outfields,mask,tt):
        if self.runargs.sanity:
            flushed_print(prob_outputs(outputs,outfields,mask))
        if self.runargs.disp > 0 and tt%self.runargs.disp==0:
            flushed_print('\t\t\t train-loss: ',str(np.mean(np.array(self.logs['train-loss'][-1]))),\
                    '\t ±',\
                    str(np.std(np.array(self.logs['train-loss'][-1]))))
            flushed_print(self.timer)

    def val_init(self,):
        self.net.eval()
        return dict(
            val_loss = 0.,
            num_val = 0.
        )
    def val_inner_loop(self,infields,outfields,mask,mem):
        with torch.set_grad_enabled(False):
            outputs = self.net.forward(infields)
            outfields,mask = outfields.to(self.device),mask.to(self.device)
            loss = self.criterion(outputs, outfields, mask)
        mem['val_loss']+=loss.item()
        mem['num_val']+=1
        return mem


    def end_of_epoch(self,mem):
        self.logs['val-loss'].append(mem['val_loss']/mem['num_val'])
        self.logs['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.scheduler.step(self.logs['val-loss'][-1])
        if len(self.logs['epoch'])>0:
            self.logs['epoch'].append(self.epoch)
        else:
            self.logs['epoch'].append(0)
        flushed_print('#epoch ',str(self.logs['epoch'][-1]),' ',\
                    ' val-loss: ',str(self.logs['val-loss'][-1]),\
                    ' train-loss: ',str(np.mean(np.array(self.logs['train-loss'][-1]))),\
                    ' lr: ',str(self.logs['lr'][-1]))
        self.state_dict = update_statedict(self.state_dict,self.net,self.optimizer,self.scheduler,last_model = True)
        if np.amin(self.logs['val-loss']) == self.logs['val-loss'][-1]:
            self.state_dict = update_statedict(self.state_dict,self.net,self.optimizer,self.scheduler,last_model = False)
        save_statedict(self.modelid,self.state_dict,self.logs)
        return self.logs['lr'][-1]<1e-7
    def train_init(self,):
        self.logs['train-loss'].append([])
        self.net.train()
    def run(self,):
        for epoch in range(self.runargs.epoch,self.runargs.maxepoch):
            self.epoch = epoch
            self.train_init()
            for infields,outfields,mask,tt in self.dataset_iterator(mode = 'train'):
                outputs = self.train_inner_loop(infields,outfields,mask)
                self.disp(outputs,outfields,mask,tt)
            mem = self.val_init()
            for infields,outfields,mask,tt in self.dataset_iterator(mode = 'val'):
                mem = self.val_inner_loop(infields,outfields,mask,mem)
            if self.end_of_epoch(mem):
                break
    