import os
cg3306 = '/scratch/cg3306/climate'
cm2p6 = '/scratch/zanna/data/cm2.6'
repo_name = 'CM2P6Param'
GRID_INFO = os.path.join(cm2p6,'GFDL_CM2_6_grid.nc')
REPO = os.path.join(cg3306,repo_name)
SLURM = os.path.join(REPO,'jobs')

MODELIDS_JSON = os.path.join(REPO,'modelids.json')
BACKUP_MODELIDS_JSON = os.path.join(REPO,'backup_modelids.json')
SAVES = os.path.join(REPO,'saves')

SLURM_LOGS = os.path.join(SAVES,'slurm_logs')
EVALS = os.path.join(SAVES,'evals')
VIEWS = os.path.join(SAVES,'views')

SCALARS = os.path.join(SAVES,'scalars')
LSRP = os.path.join(SAVES,'lsrp')
PLOTS = os.path.join(SAVES,'plots')
VIEW_PLOTS = os.path.join(PLOTS,'views')
R2_PLOTS = os.path.join(PLOTS,'r2')



MODELS = os.path.join(SAVES,'models')
TRAINING_LOGS = os.path.join(SAVES,'training_logs')
MODELS_JSON = os.path.join(SAVES,'models_info.json')
DATA_JSON = os.path.join(SAVES,'data_info.json')


for dir in [MODELS,TRAINING_LOGS,SAVES,EVALS,VIEWS,SLURM_LOGS]:
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_view_path(modelid):
    return os.path.join(VIEWS,modelid + '.nc')
def get_eval_path(modelid):
    return os.path.join(EVALS,modelid + '.nc')
def modelsdict_path():
    return MODELS_JSON
def statedict_path(modelid):
    return os.path.join(MODELS,f"{modelid}.pth")
def model_logs_json_path(modelid):
    return os.path.join(TRAINING_LOGS,f"{modelid}.json")

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

def average_lowhres_fields_path(sigma:int,isdeep):
    if isdeep:
        return os.path.join(LSRP,f'average_lowhres_{sigma}_3D.nc')
    else:
        return os.path.join(LSRP,f'average_lowhres_{sigma}_surface.nc')

def average_highres_fields_path(isdeep):
    if isdeep:
        return os.path.join(LSRP,f'average_highres_3D.nc')
    else:
        return os.path.join(LSRP,f'average_highres_surface.nc')
