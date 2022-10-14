import json
import os
from utils.arguments import options
from params import MODEL_PARAMS, get_default
import numpy as np
from utils.paths import get_eval_path, get_view_path, model_logs_json_path, statedict_path


def is_viewed(modelid):
    return os.path.exists(get_view_path(modelid))

def is_evaluated(modelid):
    return os.path.exists(get_eval_path(modelid))
def is_trained(modelid):
    statedictfile,logfile = statedict_path(modelid),model_logs_json_path(modelid)
    if not os.path.exists(statedictfile) or not os.path.exists(logfile):
        return False
    with open(logfile,'r') as f:
        logs = json.load(f)
    if logs['lr'][-1] < 1e-7 or logs['epoch'][-1] >= get_default('maxepoch') :
        return True
    return False

def find_best_match(incargs:str,):
    def read_params(args:str):
        margs,_ = options(args.split(' ') ,key="model")
        vals = {}
        listargs = args.split(' ')
        for i,x in enumerate(listargs):
            if '--' in x:
                x = x.replace('--','')
                val = margs.__getattribute__(x)
                vals[x] = val
        return vals
    def compare_strct_prms(prm,inc,):
        flag = True
        for key in MODEL_PARAMS:
            if key in inc:
                if MODEL_PARAMS[key]["type"] != float:
                    flag = inc[key]==prm[key]
            if not flag:
                return False
        return flag
    def compare_float_prms(prm,inc,):
        distance = 0
        for key in MODEL_PARAMS:
            if MODEL_PARAMS[key]["type"] != float:
                continue
            if key in inc:
                distance += abs(inc[key]-prm[key])
        return distance
    inc = read_params(incargs)
    with open(modelsdict) as f:
        models = json.load(f)
    mids = []
    for mid,args in models.items():
        prms = read_params(args)
        if compare_strct_prms(prms,inc):
            mids.append(mid)
    if len(mids)==0:
        return None,None
    distances = []
    for mid in mids:
        args = models[mid]
        prms = read_params(args)
        distances.append(compare_float_prms(prms,inc))
    i = np.argmin(distances)
    mid = mids[i]
    return models[mid],mid
