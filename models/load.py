import json
import os
from models.bank import init_architecture
from models.lossfuns import MSE, heteroscedasticGaussianLoss
import torch
from utils.arguments import options
from utils.parallel import get_device
from utils.paths import model_logs_json_path, modelsdict_path, statedict_path

def update_statedict(state_dict_,net_,optimizer_,scheduler_,last_model = True):
    if state_dict_ is None:
        state_dict_ = {}
    if last_model:
        state_dict_["last_model"] = net_.state_dict()
    else:
        state_dict_["best_model"] = net_.state_dict()
    state_dict_["optimizer"] = optimizer_.state_dict()
    state_dict_["scheduler"] = scheduler_.state_dict()
    return state_dict_



def get_statedict(modelid):
    statedictfile =  statedict_path(modelid)
    logfile = model_logs_json_path(modelid)
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

def load_modelsdict():
    file = modelsdict_path()
    if os.path.exists(file):
        with open(file) as f:
            modelsdict = json.load(f)
    else:
        modelsdict = {}
    return modelsdict


def load_model(args):
    archargs,_ = options(args,key = "arch")
    net = init_architecture(archargs)
    modelargs,modelid = options(args,key = "model")
    state_dict,logs = get_statedict(modelid)
    if modelargs.lossfun == "heteroscedastic":
        criterion = heteroscedasticGaussianLoss
    elif modelargs.lossfun == "MSE":
        criterion = MSE
    runargs,_ = options(args,key = "run")
    # if state_dict is None:
    #     warmuptime = 50
    #     strlr = 1e-8
    #     optimizer = torch.optim.Adam(net.parameters(), lr=strlr)
    #     curlr = runargs.lr
    #     gamma = torch.exp(torch.log(torch.tensor([curlr/strlr]))/warmuptime).item()
    # else:
    optimizer = torch.optim.Adam(net.parameters(), lr=runargs.lr)
        # gamma = 1.

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2)
    rerun_flag = runargs.rerun and runargs.mode == 'train'
    if state_dict is not None and not rerun_flag:
        if runargs.mode == "train":
            net.load_state_dict(state_dict["last_model"])
            net.train()
            print(f"Loaded the existing model")
        elif runargs.mode == "eval":
            net.load_state_dict(state_dict["best_model"])
            net.eval()
        print(f"Loaded the existing model")
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict:
            scheduler.load_state_dict(state_dict["scheduler"])
    else:
        if state_dict is not None:
            print(f"Model was not found")
        elif rerun_flag:
            print(f"Model is re-initiated for rerun")
    if runargs.relog:
        logs = {"epoch":[],"train-loss":[],"test-loss":[],"val-loss":[],"lr":[],"batchsize":[]}
    if len(logs["epoch"])>0:
        epoch = logs["epoch"][-1]
    else:
        epoch = 0
    runargs.epoch = epoch
    return modelid,state_dict,net,criterion,optimizer,scheduler,logs,runargs
