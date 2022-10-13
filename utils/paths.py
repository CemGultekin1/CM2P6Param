import os
cg3306 = '/scratch/cg3306/climate'
cm2p6 = '/scratch/zanna/data/cm2.6'
repo_name = 'CM2P6Param'
REPO = os.path.join(cg3306,repo_name)
SLURM = os.path.join(REPO,'slurm')
LOGS = os.path.join(SLURM,'logs')
MODELIDS_JSON = os.path.join(REPO,'modelids.json')
BACKUP_MODELIDS_JSON = os.path.join(REPO,'backup_modelids.json')
SAVES = os.path.join(REPO,'saves')

SCALARS_JSON = os.path.join(SAVES,'scalars.json')
LSRP = os.path.join(SAVES,'lsrp')




MODELS_DIR = os.path.join(SAVES,'models')
MODEL_LOGS_DIR = os.path.join(SAVES,'logs')
MODELS_JSON = os.path.join(SAVES,'models_info.json')
DATA_JSON = os.path.join(SAVES,'data_info.json')

for dir in [MODELS_DIR,MODEL_LOGS_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)

def modelsdict_path():
    return MODELS_JSON
def statedict_path(modelid):
    return os.path.join(MODELS_DIR,f"{modelid}.pth")
def model_logs_json_path(modelid):
    return os.path.join(MODEL_LOGS_DIR,f"{modelid}.json")

def search_compressed_lsrp_paths(sigma:int,):
    fns = os.listdir(LSRP)
    fns = [fn for fn in fns if f'compressed_conv_weights_{sigma}' in fn]
    spns = []
    for fn in fns:
        lstprt = fn.split('_')[-1]
        span_ = int(lstprt.split('.')[0])
        spns.append(span_)
    return fns,spns
def convolutional_lsrp_weights_path(sigma:int,span :int = -1):
    if span < 0:
        return os.path.join(LSRP,f"conv_weights_{sigma}.nc")
    else:
        return os.path.join(LSRP,f'compressed_conv_weights_{sigma}_{span}.nc')

def inverse_coarse_graining_weights_path(sigma:int):
    return os.path.join(LSRP,f'inv_weights_{sigma}.nc')
def coarse_graining_projection_weights_path(sigma:int):
    return os.path.join(LSRP,f'proj_weights_{sigma}.nc')
