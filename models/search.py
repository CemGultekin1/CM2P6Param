import json
from utils.arguments import options
from models.paths import modelsdict
from params import MODEL_PARAMS
import numpy as np
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


    


