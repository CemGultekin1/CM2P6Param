import json
from utils.parallel import random_wait
from utils.paths import BACKUP_MODELIDS_JSON, MODELIDS_JSON

def reset():
    data_info={}
    lookupfile=MODELIDS_JSON
    with open(lookupfile,'w') as infile:
        json.dump(data_info,infile)

def safe():
    lookupfile=MODELIDS_JSON
    with open(lookupfile) as infile:
        data_info=json.load(infile)

    lookupfile=BACKUP_MODELIDS_JSON
    with open(lookupfile,'w') as infile:
        json.dump(data_info,infile)
def recover():
    lookupfile=BACKUP_MODELIDS_JSON
    random_wait()
    with open(lookupfile) as infile:
        data_info=json.load(infile)

    lookupfile=MODELIDS_JSON
    random_wait()
    with open(lookupfile,'w') as infile:
        json.dump(data_info,infile)

def get_dict(model_bank_id,model_id,parallel=True):
    expand_dict(model_bank_id,model_id)
    lookupfile=MODELIDS_JSON
    random_wait(parallel=parallel)
    with open(lookupfile) as infile:
        data_info_=json.load(infile)
    return data_info_[str(model_bank_id)][str(model_id)]
def expand_dict(model_bank_id,model_id,parallel=True):
    lookupfile=MODELIDS_JSON
    random_wait(parallel=parallel)
    with open(lookupfile) as infile:
        data_info_=json.load(infile)
    iskey1=False
    iskey2=False
    if str(model_bank_id) in list(data_info_.keys()):
        iskey1=True
        if str(model_id) in list(data_info_[str(model_bank_id)].keys()):
            iskey2=True

    if not iskey1:
        data_info_[str(model_bank_id)]={}
    if not iskey2:
        data_info_[str(model_bank_id)][str(model_id)]={}
    random_wait(parallel=parallel)
    with open(lookupfile,'w') as infile:
        json.dump(data_info_,infile)
def update_model_info(data_info,model_bank_id,model_id,parallel=True):
    expand_dict(model_bank_id,model_id)
    lookupfile=MODELIDS_JSON
    random_wait(parallel=parallel)
    with open(lookupfile) as infile:
        data_info_=json.load(infile)
    data_info_[str(model_bank_id)][str(model_id)]=data_info.copy()
    random_wait(parallel=parallel)
    with open(lookupfile,'w') as infile:
        json.dump(data_info_,infile)
