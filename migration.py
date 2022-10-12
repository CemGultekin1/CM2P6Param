from copy import deepcopy
import json
from typing import OrderedDict
from models.nets.cnn import CNN
from models.save import save_statedict, update_modelsdict
from utils.old_bank import golden_model_bank
from utils.arguments import options
from argparse import Namespace
import torch
import os
# import torch

root = '/scratch/cg3306/climate/'
source = os.path.join(root,'runs')
logstarget = os.path.join(root,'saves','logs')
modelstarget = os.path.join(root,'saves','models')


def model_iterator(testid:int):
    modeldirs = os.listdir(source)
    STEP=1000
    for m in modeldirs:
        if 'G-' not in m:
            continue
        modelid = int(m[m.find('G-') + len('G-'):])
        testflag = modelid//STEP - testid == 0
        if not testflag:
            continue
        yield os.path.join(root,source,m),modelid
def namespace2str(ns:Namespace):
    st = []
    for key,val in ns.__dict__.items():
        if isinstance(val,list):
            valst = ' '.join([str(v) for v in val])
        else:
            valst = str(val)
        st.append(f'--{key} {valst}')
    return ' '.join(st)
def layer_state_dict(state_dict):
    print(state_dict.keys())
    sd = OrderedDict()
    for key,val in state_dict.items():
        layerid = int(key.split('.')[1])
        if layerid not in sd:
            sd[layerid] = OrderedDict()
        sd[layerid][key.split('.')[-1]] = val
    return sd
def pass_weights(cnn:CNN,state_dict:OrderedDict):
    i=0
    # print(state_dict)
    # print(cnn)
    print(state_dict.keys())
    print(cnn)
    raise Exception('hi')
    for lyr in state_dict.values():
        batchnormflag = "running_mean" in lyr
        if not batchnormflag:
            cnn._modules['conv_body']._modules[f"conv-{i}"]._modules['kernel'].weight.data = lyr['weight']
            cnn._modules['conv_body']._modules[f"conv-{i}"]._modules['kernel'].bias.data = lyr['bias']
        else:
            cnn._modules['conv_body']._modules[f"conv-{i}"]._modules['batchnorm'].weight.data = lyr['weight']
            cnn._modules['conv_body']._modules[f"conv-{i}"]._modules['batchnorm'].bias.data = lyr['bias']
            i+=1
def create_state_dict(modelid,archargs,dir):
    logpath = os.path.join(dir,'log.json')
    if os.path.exists(logpath):
        with open(logpath) as f:
            logs = json.load(f)
    else:
        logs = None
    new_state_dict = {}
    for name in ['best-model','last-model']:
        model_path = os.path.join(dir,name)
        # if '/scratch/cg3306/climate/runs/G-76/' not in model_path:
        #     continue
        # print(model_path)
        if not os.path.exists(model_path):
            continue
        # cnn = CNN(**archargs.__dict__)
        # print(model_path)
        # state_dict = layer_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        new_state_dict[name] = torch.load(model_path,map_location=torch.device('cpu'))
        # pass_weights(cnn,state_dict)
        # try:
        #     net1=CNN(**archargs.__dict__)
        #     net1.load_state_dict(cnn.state_dict())#modelid,)#new_state_dict[name])
        #     new_state_dict[name] = deepcopy(cnn.state_dict())
        # except:
        #     print('failed to load state dict of :',modelid)

    if len(new_state_dict) == 0 :
        new_state_dict = None
    return new_state_dict,logs

def get_target_models(testid):
    models = {}
    for loaddir,model_id in model_iterator(testid):
        args = Namespace()
        args.model_id=model_id
        args.co2 = 0

        newargs,description = golden_model_bank(args)
        if newargs is None:
            continue

        args = namespace2str(newargs)
        _, modelid = options(args.split(),key = "model")
        models[modelid] = (args,loaddir)
    return models
def migrate_models(models:dict):
    for modelid,(args,loaddir) in models.items():
        # if modelid != '68a0729fdc1521b592a82effab4ea888bbf61ff16c94cd7a3de48ea8':
            # continue
        modelargs, modelid = options(args.split(),key = "model")
        archargs, _ = options(args.split(),key = "arch")
        state_dict,logs = create_state_dict(modelid,archargs,loaddir)
        # if state_dict is not None:
        #     print(modelargs,modelid)
        print(modelid)
        save_statedict(modelid,state_dict,logs)
        update_modelsdict(modelid,args)
        # break


def delete_models(testid):
    modelsdict = get_target_models(testid)
    for fname in modelsdict.keys():
        for d,exc in zip([modelstarget,logstarget],['.pth','.json']):
            file = os.path.join(d,fname+exc)
            if os.path.exists(file):
                os.remove(file)


allids = {  0: 'depth-generalization',
            9: 'resnet-generalization',
            8: 'filter-sizes',
            7: 'shrinkage-types',
            3: 'improvements',
            5: 'improvements-and-models',
            1: 'nontemperature-for-distribution',
            }

doneids = [0,8,7,3,]
newids = [3,]
for testid in newids:
    modelsdict = get_target_models(testid)
    migrate_models(modelsdict)
