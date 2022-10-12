import itertools
import os
import pickle
from typing import List
from data.datasets import Dataset
from data.load import get_loaders, load_normalization_scalars
from models.load import load_from_save
from plots.projections import imshow_plot, line_plots
from utils.arguments import options
from utils.parallel import get_device
import torch
import numpy as np
from models.search import find_best_match

def pre_processing(datargs,recfield,*datasets:List[Dataset]):
    load_normalization_scalars(datargs,datasets[0])
    assert datasets[0].inscalars is not None
    datasets[0].set_receptive_field(recfield)
    for i in range(1,len(datasets)):
        datasets[i].receive_scalars(datasets[0])
    return datasets

def show_training():
    data = []
    titles = []
    kwargs = []
    def change_data(margs,*args):
        marglist = margs.split(' ')
        for i in range(len(args)//2):
            key,val = args[i*2],args[i*2+1]
            i = marglist.index(f'--{key}')+1
            marglist[i] = str(val)
        return ' '.join(marglist)
    codes = [[1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0]]
    # codes = [[1, 1, 0, 0, 1, 0],
    #         [0, 1, 0, 0, 1, 0],
    #         [1, 1, 0, 1, 1, 0],
    #         [1, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0]]
    sigma = 4
    # linsupres = [False,True]
    # lats = [False,True]
    for code in codes:
        temp = bool(code[0])
        dom = "four_regions" if not code[1] else "global"
        linsup = bool(code[2])
        lat = bool(code[3])
        print(code,(temp,dom,linsup,lat))
        modelname = ''
        if dom == "global":
            modelname +='G'
        if temp:
            modelname +='T'
        if lat:
            modelname += 'L'
        if linsup:
            modelname += 'S'
        # if not temp:
        #     continue

        modelname += '-CNN'
        modelargs = f'--domain {dom} --temperature {temp} --latitude {lat} --sigma {sigma} --depth 0 --linsupres {linsup}'
        modelargs_,modelid = find_best_match(modelargs)
        if modelargs_ is None:
            print('NOT FOUND: ',modelname)
            continue
        else:
            modelargs = modelargs_
        _,_,_,_,_,_,logs,_=load_from_save(modelargs.split(' '))
        trainloss = np.array([np.mean(a) for a in logs['train-loss']])
        valloss = np.array(logs['val-loss'])
        data.append((trainloss,valloss))
        titles.append(modelname)
        kwargs.append({"color":["b","r"],"label":["train","val"]})

    # nmodels = len(data)
    nrow = 2
    ncol = 3
    figsize = (nrow*6,ncol*6)
    suptitle = ""
    rootdir = '/scratch/cg3306/climate/saves/plots/'
    plotroot = os.path.join(rootdir,'surf-snapshots-1')
    if not os.path.exists(plotroot):
        os.makedirs(plotroot)
    filename = os.path.join(plotroot,f"training.png")
    line_plots(data,titles,nrow,ncol,figsize,suptitle,filename,kwargs)


def run_models(datamodel:dict):
    nets = {}
    datasets = {}
    for dataid,idlist in datamodel.items():
        for datargs,modelargs,modelid,modelname in idlist:
            modelid,_,net,_,_,_,_,_=load_from_save(modelargs.split(' '))
            
            nets[modelid] = net
        (dataset,_),_,_=get_loaders(datargs.split(' '))
        '''
        "xmin":[-50,-180,-110,-48],
                "xmax":[-20,-162,-92,-30],
                "ymin":[35,-40,-20,0],
                "ymax":[50,-25,-5,15]
        '''
        # pre_processing(datargs,net.receptive_field,dataset)
        # ext = dataset.most_ocean_points(30,30,5,1)[0]
        extent = [-160.35, -120.35000000000136, 20.412099623311953 ,48.724758482833984]
        dataset.confine(0,*extent)
        

        
        # print(ext)
        # dataset.confine(0,*ext)
        # dataset.outmasks[0] = None
        pre_processing(datargs,net.receptive_field,dataset)

        # print(dm)
        dataset.set_same_padding(False)
        n = len(dataset)
        datasets[dataid] = dataset

    device=get_device()
    
    stak = lambda G: torch.from_numpy(np.stack([G],axis=0)).type(torch.float32).to(device)
    nonstak = lambda G: G[0].numpy()[:,::-1]
    outputs = {}
    np.random.seed(0)
    indices = np.sort(np.random.choice(np.arange(n),size =16, replace=False))
    for index in indices:
        for dataid,dataset in datasets.items():
            for (datargs,modelargs,modelid,modelname) in datamodel[dataid]:
                print(dataid,modelid,index)
                net = nets[modelid]
                dataset = datasets[dataid]
                
                X,Y,mask = (stak(G) for G in dataset[index])
                net.eval()
                with torch.set_grad_enabled(False):
                    mean,_ = net.forward(X)
                # print(X.shape,Y.shape,mask.shape,mean.shape)
                mask[mask==0] = np.nan
                Y = Y*mask
                mean = mean*mask

                X,Y,mask,mean = (nonstak(G) for G in (X,Y,mask,mean))
                mean = Y[:mean.shape[0]] - mean

                X = dataset.to_physical(X,input=True)
                Y = dataset.to_physical(Y,input=False)
                mean = dataset.to_physical(mean,input=False)

                if X.shape[0]==2:
                    X = np.concatenate([X,X[:1]*np.nan],axis=0)
                    mean = np.concatenate([mean,mean[:1]*np.nan],axis=0)
                    Y = None
                
                if modelargs not in outputs:
                    outputs[modelargs] = []
                fields = (("u","v","T"),("Su","Sv","ST"))
                coords = dataset.coords[0]
                # pprints = ["X:",X.shape,"\tY:",Y.shape,"\tmean:",mean.shape,\
                #     "\tlat:",coords[0].shape,"\tfields:",fields,"\t",modelname]
                # pprints = [str(p) for p in pprints]
                # print(' '.join(pprints))
                outputs[modelargs].append((X,Y,mean,coords,fields,modelname,index))
    rootdir = '/scratch/cg3306/climate/saves/plots/data'
    filename = 'snapshot_data.pkl'
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    path = os.path.join(rootdir,filename)
    with open(path , 'wb') as f:
        pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)
    
def plot_snapshots():
    rootdir = '/scratch/cg3306/climate/saves/plots/data'
    filename = 'snapshot_data.pkl'
    path = os.path.join(rootdir,filename)
    with open(path , 'rb') as f:
       output = pickle.load(f)
    
    nsnapshots = len(output[list(output.keys())[0]])
    def plot_snapshot(output,si):
        
        nmodels = len(output)+3
        def init_params(nchans,nmodels):
            data = np.empty((nchans,nmodels),dtype = object)
            lons = np.empty((nchans,nmodels),dtype = object)
            lats = np.empty((nchans,nmodels),dtype = object)
            titles = np.empty((nchans,nmodels),dtype = object)
            kwargs = np.empty((nchans,nmodels),dtype = object)
            nrow = nchans
            ncol = nmodels
            figsize = (ncol*6,nrow*4)
            return data,lons,lats,titles,nrow,ncol,figsize,kwargs
        data = None
        modelnames = []
        lsrpflag = False
        for j,key in enumerate(list(output.keys())):
            X,Y,mean ,(lat,lon),fields,modelname,index= output[key][si]
            # X,Y,mean = (A[:,100:200,100:200] for A in (X,Y,mean))
            modelnames.append(modelname)
            if data is None:
                nchans = 3#X.shape[0]
                data,lons,lats,titles,nrow,ncol,figsize,kwargs = init_params(nchans,nmodels)
                for i in range(nchans):
                    data[i,0] = X[i] # inputs
                spread = (Y[0].shape[0] - X[0].shape[0])//2
                for i in range(nchans):
                    data[i,1] = Y[i] # true outputs
            for i in range(nchans):
                data[i,j+2] = mean[i] # prediction
            if Y is None:
                continue
            if Y.shape[0] == 2*nchans: # if it is residual
                lsrpflag = True
                lsrpoutput = Y[nchans:] - Y[:nchans]
                for i in range(nchans):
                    data[i,j+2] += lsrpoutput[i]
                for i in range(nchans):
                    data[i,-1] = lsrpoutput[i]
        if lsrpflag:
            modelnames.append("LSRP")
        else:
            nmodels-=1
            _,_,_,_,nrow,ncol,figsize,kwargs = init_params(nchans,nmodels)
            cutlast  = lambda vecs: [vec[:,:-1] for vec in vecs]
            data,lons,lats,titles = cutlast((data,lons,lats,titles))

        for i,j in itertools.product(range(nchans),range(1)):
            titles[i,j] = fields[0][i]
            lons[i,j] = lon
            lats[i,j] = lat
            kwargs[i,j] = {'units':'m/s' if i<2 else '$C^{o}$'}
        for i,j in itertools.product(range(nchans),range(1,2)):
            titles[i,j] = fields[1][i]
            lons[i,j] = lon[spread:-spread]
            lats[i,j] = lat[spread:-spread]
            kwargs[i,j] = {'units':'x$10^{-7}$ m/$s^2$'}
        for i,j in itertools.product(range(nchans),range(2,nmodels)):
            titles[i,j] = f"{modelnames[j-2]}: {fields[1][i]}"
            lons[i,j] = lon[spread:-spread]
            lats[i,j] = lat[spread:-spread]
            kwargs[i,j] = {'units':'x$10^{-7}$ m/$s^2$'}
        suptitle = f"Models Compared on 0m Depth Dataset, Time Index = {index}"
        flatten = lambda vec : vec.reshape([-1])

        vmin = [np.inf]*nchans
        vmax = [-np.inf]*(nmodels-1)
        for i,j in itertools.product(range(nchans),range(2,nmodels)):
            x = data[i,j]
            x = x[x==x]
            if j==1:
                vmax[i] = np.maximum(vmax[i],np.amax(x))
                vmin[i] = np.minimum(vmin[i],np.amin(x))
        for i in range(nchans):
            if vmin[i]<=0 and vmax[i] >=0:
                ext = np.maximum(np.abs(vmin[i]),np.abs(vmax[i]))
                vmin[i] = -ext
                vmax[i] = ext
        for i,j in itertools.product(range(nchans),range(1,nmodels)):
            kwargs[i,j]["vmax"] = vmax[i]
            kwargs[i,j]["vmin"] = vmin[i]
        extent = [-160.35, -120.35000000000136, 20.412099623311953 ,48.724758482833984]
        for i,j in itertools.product(range(nchans),range(nmodels)):
            kwargs[i,j]["extent"] = extent
        data,lons,lats,titles,kwargs = \
            (flatten(x) for x in (data,lons,lats,titles,kwargs))
        rootdir = '/scratch/cg3306/climate/saves/plots/'
        plotroot = os.path.join(rootdir,'surf-snapshots-1')
        if not os.path.exists(plotroot):
            os.makedirs(plotroot)
        filename = os.path.join(plotroot,f"snapshot-{si}.png")
        imshow_plot(data,lons,lats,titles,nrow,ncol,figsize,suptitle,filename,kwargs)
    for si in range(nsnapshots):
        plot_snapshot(output,si)
    
            
    
def save_snapshots():
    def change_data(margs,*args):
        marglist = margs.split(' ')
        for i in range(len(args)//2):
            key,val = args[i*2],args[i*2+1]
            i = marglist.index(f'--{key}')+1
            marglist[i] = str(val)
        return ' '.join(marglist)

    datamodel = {}
    codes = [[1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0]]
    # codes = [[1, 1, 0, 0, 1, 0],
    #         [0, 1, 0, 0, 1, 0],
    #         [1, 1, 0, 1, 1, 0],
    #         [1, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0]]
    sigma = 4
    # linsupres = [False,True]
    # lats = [False,True]
    for code in codes:
        temp = bool(code[0])
        dom = "four_regions" if not code[1] else "global"
        linsup = bool(code[2])
        lat = bool(code[3])
        print(code,(temp,dom,linsup,lat))
        modelname = ''
        if dom == "global":
            modelname +='G'
        if temp:
            modelname +='T'
        if lat:
            modelname += 'L'
        if linsup:
            modelname += 'S'
        # if not temp:
        #     continue

        modelname += '-CNN'
        modelargs = f'--domain {dom} --temperature {temp} --latitude {lat} --sigma {sigma} --depth 0 --linsupres {linsup}'
        modelargs_,modelid = find_best_match(modelargs)
        if modelargs_ is None:
            print('NOT FOUND: ',modelname)
            continue
        else:
            modelargs = modelargs_
        print(modelname)
        datargs = change_data(modelargs,'domain','global')
        
        _,dataid = options(modelargs.split(' '),key = "data")
        # print(datargs_,'\n') 
        if dataid not in datamodel:
            datamodel[dataid] = []
        datamodel[dataid].append((datargs,modelargs,modelid,modelname))
    run_models(datamodel) 

def main():
    save_snapshots()
    plot_snapshots()
    # show_training()

 
if __name__=='__main__':
    main()
 