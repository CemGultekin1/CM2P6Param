from typing import List
from argparse import ArgumentTypeError

USUAL_PARAMS = {
    "depth" : (0,5,55,110,181,330,1497),
    "sigma" : (4,8,12,16),
}

SCALAR_PARAMS = {
    "sigma" : {"type": int, "choices" : (4,8,12,16)},
    "depth" : {"type": float, "default" : 0.},
    "co2" : {"type":bool,"default":False},
}



DATA_PARAMS = {
    "domain" : {"type": str, "choices" : ["four_regions","global","custom"],},
    "temperature" : {"type": bool, "default":False},
    "latitude" : {"type": bool, "default":False},
    "lsrp" :  {"type": int, "default":0},
    "parts": {"type":int, "nargs": 2, "default":(1,1)},
}




TRAIN_PARAMS = {
    "lr" : {"type": float, "default" : 1e-2},
    "lossfun" : {"type":str, "choices":["heteroscedastic","MSE"]}
}


EVAL_PARAMS = {
    "modelid" : {"type":str,"default" : ""},
    "dataids" : {"type":str, "nargs":'+',"default" : ""}
}






ARCH_PARAMS = {
    "kernels" : {"type": int, "nargs":'+', "default" : (5,5,3,3,3,3,3,3)},
    "widths" : {"type": int,  "nargs":'+',"default" : (2,128,64,32,32,32,32,32,4)},
    "skipconn" : {"type":int,"nargs":'+',"default":tuple([0]*8)},
    "batchnorm" : {"type":int,"nargs":'+',"default":tuple([1]*8)},
    "seed" : {"type":int,"default":0},
}



RUN_PARAMS = {
    "minibatch" : {"type": int, "default" : 1},
    "num_workers" : {"type": int, "default" : 0},
    "prefetch_factor" : {"type": int, "default": 1},
    "maxepoch" : {"type": int, "default" : 500},
    "persistent_workers" : {"type":bool,"default":True},
    "rerun":{"type":bool,"default":False},
    "relog":{"type":bool,"default":False},
    "disp" :  {"type":int,"default":-1},
    "mode" : {"type": str, "choices" : ["train","eval","data","scalars","view"],},
    "section" : {"type":int, "nargs": 2, "default":(0,1)},
    "sanity": {"type":bool, "default":False},
    "lsrp_span": {"type":int,  "default": 12},
}



PARAMS = dict(TRAIN_PARAMS,**DATA_PARAMS,**RUN_PARAMS,**ARCH_PARAMS,**SCALAR_PARAMS)

DATA_PARAMS = dict(DATA_PARAMS,**SCALAR_PARAMS)
TRAIN_PARAMS = dict(TRAIN_PARAMS,**DATA_PARAMS)
MODEL_PARAMS = dict(TRAIN_PARAMS,**ARCH_PARAMS)
RUN_PARAMS = dict(TRAIN_PARAMS,**RUN_PARAMS)




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')




for d in (DATA_PARAMS,ARCH_PARAMS,RUN_PARAMS,SCALAR_PARAMS,TRAIN_PARAMS,PARAMS):
    for key in d:
        if "choices" in d[key]:
            d[key]["default"] = d[key]["choices"][0]
        if d[key]["type"]==bool:
            d[key]["dest"] =key
            d[key]["type"] = str2bool

def repr(tpl):
    if tpl is None:
        return ""
    if isinstance(tpl,tuple):
        return " ".join([str(x) for x  in tpl])
    else:
        return str(tpl)

def defaulting_dict(d:dict,key:str,**kwargs):
    return d.get(key,get_default(key,**kwargs))

def get_default(key,instr = False):
    if key not in PARAMS:
        return_val = None
    else:
        return_val = PARAMS[key]["default"]
    if instr:
        return repr(return_val)
    else:
        return return_val


def replace_param(args,param,newval):
    if not isinstance(newval,str):
        newval = repr(newval)
    param_ = f'--{param}'
    if param_ in args:
        args[args.index(param_)+1] = newval
    else:
        args.append(param_)
        args.append(newval)
    return args