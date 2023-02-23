import os

from utils.arguments import options
from utils.paths import GRID_INFO,CM2P6_PATH,COARSE_CM2P6_PATH,TEMPORARY_DATA



def get_filename(sigma,depth,co2,filtering,locdir = CM2P6_PATH):
    if sigma > 1:
        co2 = '1pct_co2' if co2 else ''
        surf = 'surface' if depth < 1e-3 else 'beneath_surface'
        filename = f'coarse_{sigma}_{surf}_{co2}.zarr'
        filename = filename.replace('_.zarr','.zarr')
    else:
        co2 = '1pct_co2' if co2 else ''
        surf = 'surface' if depth < 1e-3 else 'beneath_surface'
        filename = f'{surf}_{co2}.zarr'
        filename = filename.replace('_.zarr','.zarr')
    if sigma > 1:
        locdir = COARSE_CM2P6_PATH
    path = os.path.join(locdir,filename)
    
    if sigma > 1:
        if filtering is not None:
            path = path.replace('.zarr',f'_{filtering}.zarr')
    
    return path

def get_high_res_grid_location():
    return GRID_INFO

def get_high_res_data_location(args):
    prms,_ = options(args,key = "data")
    return get_filename(1,prms.depth,prms.co2,prms.filtering)

def get_low_res_data_location(args):
    prms,_ = options(args,key = "data")
    return  get_filename(prms.sigma,prms.depth,prms.co2,prms.filtering)

def get_preliminary_low_res_data_location(args):
    prms,_ = options(args,key = "run")
    a,b = prms.section
    filename = get_filename(prms.sigma,prms.depth,prms.co2,prms.filtering,locdir = TEMPORARY_DATA)
    filename = filename.replace('.zarr',f'_{a}_{b}.zarr')
    return filename

def get_data_address(args):
    drs = os.listdir(COARSE_CM2P6_PATH)
    def searchfor(drs,tokens):
        for token in tokens:
            drs = [f for f in drs if token in f]
        return drs
    def filterout(drs,tokens):
        for token in tokens:
            drs = [f for f in drs if token not in f]
        return drs

    prms,_ = options(args,key = "data")
    includestr = '.zarr'
    excludestr = 'coarse'
    if prms.co2:
        includestr += ' CO2'
    else:
        excludestr += ' CO2'

    if prms.depth <1e-3 :
        includestr += ' surf'
    else:
        assert prms.depth > 0
        includestr += ' 3D'
    if len(excludestr)>0:
        excludestr = excludestr[1:]
    include = includestr.split()
    exclude = excludestr.split()
    drs =searchfor(drs,include)
    drs = filterout(drs,exclude)
    if len(drs) == 1:
        return os.path.join(COARSE_CM2P6_PATH,drs[0])
    else:
        return None