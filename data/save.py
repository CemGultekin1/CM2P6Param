import itertools
import json
import os
from data.exceptions import RequestDoesntExist
from data.load import depthvals, load_dataset,sigmavals,domainvals,normalizationvals
from utils.arguments import options
import multiprocessing as mp
from utils.paths import SCALARS_JSON
from utils.slurm import flushed_print

def scalars_exist(args):
    scalarns,scalarid = options(args,key = 'scalar')
    # print(scalarns,'\n\t\t',scalarid)
    file = SCALARS_JSON
    if not os.path.exists(file):
        return False,None
    else:
        with open(file) as f:
            scalars = json.load(f)
    if scalarid not in scalars:
        return False,None
    return True, scalars[scalarid]

def save_scalar(args,stats):
    _,scalarid = options(args,key = 'scalar')
    file = SCALARS_JSON
    if not os.path.exists(file):
        scalars = {scalarid:{}}
    else:
        with open(file) as f:
            scalars = json.load(f)
    if scalarid not in scalars:
        scalars[scalarid] = {}
    for key,val in stats.items():
        scalars[scalarid][key] = val
    with open(file,'w') as f:
        json.dump(scalars,f,indent='\t')

def get_scalar(args,):
    stats = load_dataset(args).compute_scalars(time_window = 50)
    return stats

def iterate_datargs():
    nsigma = len(sigmavals)
    ndepth = len(depthvals)
    ndoms = len(domainvals)
    nco2  = 2
    nnorm = len(normalizationvals)
    numdata = ndepth*nsigma*ndoms*nco2*nnorm
    
    for i in range(numdata):
        domain = domainvals[i%ndoms]
        i=i//ndoms

        sigma = sigmavals[i%nsigma]
        i=i//nsigma

        depth = depthvals[i%ndepth]
        i=i//ndepth

        co2 = i%nco2 == 1
        i = i//nco2

        normalization = normalizationvals[i%nnorm]
        
        parts = "1 1"
        if domain == "global":
            parts = "3 3"

        args_ = f'--domain {domain} --co2 {co2} --parts {parts} --depth {depth} --sigma {sigma} --normalization {normalization} --linsupres True --temperature True --nworkers 0'
        args = args_.split()
        yield args

def iterate_sub_datargs():
    depthvals=[5 , 55,  110, 181,  330, 1497, 0]
    sigmavals=[8,4,12,16]
    normalization = 'standard'
    co2s = [False, True]
    domains = ["global","four_regions"]
    for domain,co2,sigma,depth in itertools.product(domains,co2s,sigmavals,depthvals):
        parts = "1 1"
        if domain == "global":
            parts = "3 3"
        args_ = f'--domain {domain} --co2 {co2} --parts {parts} --depth {depth} --sigma {sigma} --normalization {normalization} --linsupres True --temperature True --nworkers 0'
        args = args_.split()
        yield args

def submain(args,pk,q:mp.Queue,ind):
    try:
        stats = get_scalar(args)
        q.put((ind,args,stats))
    except RequestDoesntExist:
        flushed_print('\tfailed',**pk)

def main():
    pk = dict(flush = True)
    q = mp.Queue()
    ps = []
    for i,args in enumerate(iterate_sub_datargs()):#iterate_datargs()):
        p = mp.Process(target=submain, args=(args,pk,q,i))
        ps.append([i,p])
    parallel = 8
    mps = []
    for i,p in ps:
        mps.append([i,p])
        if len(mps)==parallel:
            for j,pp in mps:
                flushed_print('job\t',j)
                pp.start()
            for j,pp in mps:
                flushed_print('waiting\t',j)
                pp.join()
            while q.qsize()>0:
                ind,args,stats = q.get()
                flushed_print('saving\t',ind)
                save_scalar(args,stats)
            mps = []

if __name__=='__main__':
    main()