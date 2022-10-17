import os
from models.load import load_modelsdict
from models.save import update_modelsdict
from models.search import is_trained
from utils.arguments import options
from utils.paths import MODELS, SLURM, model_logs_json_path, statedict_path

def read_job_txts():
    fns = os.listdir(SLURM)
    fns = [os.path.join(SLURM,f) for f in fns if '.txt' in f]
    lines = []
    def read_lines(filename):
        lines = []
        with open(filename) as file:
            for line in file:
                lines.append(line.rstrip())
        return lines
    for fn in fns:
        lines.extend(read_lines(fn))
    return lines
    
def add_trained_models():
    from jobs.trainjob import TRAINJOB
    argsfile = TRAINJOB + '.txt'
    path = os.path.join(SLURM,argsfile)
    with open(path,'r') as f:
        lines = f.readlines()
    modelsdict = load_modelsdict()
    for i in range(len(lines)):
        _,modelid = options(lines[i].split(),key = "model")
        flag = is_trained(modelid)
        if flag and modelid not in modelsdict:
            update_modelsdict(modelid,lines[i])
            # print(modelid,'\t',lines[i])
    
    


def all_trained_models():
    lines = read_job_txts()
    for line in lines:        
        runprms,_ = options(line.split(),key = "run")
        if runprms.mode != 'train':
            continue
        _,modelid = options(line.split(),key = "model")
        update_modelsdict(modelid,line)
def find_untrained_models():
    modelids = os.listdir(MODELS)
    modelids = [mid.split('.')[0] for mid in modelids]
    modelsdict = load_modelsdict()
    trained_model = []
    for modelid in modelids:
        trained_model.append(modelid in modelsdict)
    return [mid for flag,mid in zip(trained_model, modelids) if not flag]
def delete_untrained_model_dirs():
    modelids = find_untrained_models()
    for modelid in modelids:
        os.remove(statedict_path(modelid))
        os.remove(model_logs_json_path(modelid))

    
def main():
    add_trained_models()


if __name__ == '__main__':
    main()