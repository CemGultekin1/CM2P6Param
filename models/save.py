import os
import torch
import json
from models.load import load_modelsdict
from models.search import is_trained
from utils.paths import model_logs_json_path, modelsdict_path, statedict_path

def save_statedict(modelid,statedict,logs):
    statedictfile,logfile = statedict_path(modelid),model_logs_json_path(modelid)
    if statedict is not None:
        torch.save(statedict,statedictfile)
    if logs is not None:
        with open(logfile,'w') as f:
            json.dump(logs,f)

def only_fully_trained_modelsdict():
    md = load_modelsdict()
    md = {key: val for key,val in md.items() if is_trained(key)}
    overwrite_modelsdict(md)
    

def overwrite_modelsdict(md):
    file = modelsdict_path()
    with open(file,'w') as f:
        json.dump(md,f,indent='\t')

def update_modelsdict(modelid,args):
    file = modelsdict_path()
    if os.path.exists(file):
        with open(file) as f:
            modelsdict = json.load(f)
    else:
        modelsdict = {}
    if isinstance(args,list):
        args = ' '.join(args)
    modelsdict[modelid] = args
    with open(file,'w') as f:
        json.dump(modelsdict,f,indent='\t')
