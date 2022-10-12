import json
import os
import torch

from utils.parallel import get_device
from utils.paths import MODELS_JSON, SAVES, model_logs_json_path, statedict_path

root = SAVES

def get_statedict(modelid):
    statedictfile,logfile = get_statedict_file(modelid)
    device = get_device()
    if os.path.exists(statedictfile):
        print(f"model {modelid} state_dict has been found")
        state_dict = torch.load(statedictfile,map_location=torch.device(device))
        with open(logfile) as f:
            logs = json.load(f)
    else:
        print(f"model {modelid} state_dict has not been found")
        state_dict = None
        logs = {"epoch":[],"train-loss":[],"test-loss":[],"val-loss":[],"lr":[],"batchsize":[]}
    return state_dict,logs
def get_statedict_file(modelid):
    statedictfile =  statedict_path(modelid)
    logfile = model_logs_json_path(modelid)
    return statedictfile,logfile

def get_modelsdict_file():
    return MODELS_JSON
