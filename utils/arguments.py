import argparse
import hashlib
from typing import List
from params import DATA_PARAMS,MODEL_PARAMS,ARCH_PARAMS,RUN_PARAMS, SCALAR_PARAMS, TRAIN_PARAMS

def options(string_input,key:str = "model"):
    if key == "model":
        prms = MODEL_PARAMS
    elif key == "arch":
        prms = ARCH_PARAMS
    elif key == "data":
        prms = DATA_PARAMS
    elif key == "run":
        prms = RUN_PARAMS
    elif key == "scalar":
        prms = SCALAR_PARAMS
    elif key == "train":
        prms = TRAIN_PARAMS
    else:
        raise Exception('not implemented')
    

    model_parser=argparse.ArgumentParser()
    for argname,argdesc in prms.items():
        model_parser.add_argument(f"--{argname}",**argdesc)

    # model_parser.parse_known_intermixed_args
    args,_ = model_parser.parse_known_args(string_input)
    return args,args2num(prms,args)

def args2num(prms:dict,args:argparse.Namespace):
    s = []
    nkeys = len(prms)
    def append_el(l:List,i,el):
        if isinstance(el,list):
            ell = hash(tuple(el))
        elif isinstance(el,float):
            ell = int(el*1e6)
        else:
            ell = el
        l.append(i)
        l.append(ell)
        return l

    for i,(u,v) in zip(range(nkeys),args.__dict__.items()):
        if prms[u]["default"] != v:
            s = append_el(s,i,v)
            
   
    s = tuple(s)
    return hashlib.sha224(str(s).encode()).hexdigest()   