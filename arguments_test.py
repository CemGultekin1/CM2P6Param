from utils.arguments import options
import numpy as np
import os
def put_new_names():
    file = open('jobs/trainjob.txt','r')
    lines = file.readlines()
    file.close()

    keys = 'data model scalars'.split()
    oldnamefiles = {}
    for key in keys:
        filename = f'id_new_{key}.txt'
        file = open(filename,'w')
        oldnamefiles[key] = file
    
    for line in lines:
        line = line.split()
        for key in keys:
            file = oldnamefiles[key]
            _,prms_id = options(line,key = key)
            file.writelines(prms_id+'\n')
def put_old_names():
    file = open('jobs/trainjob.txt','r')
    lines = file.readlines()
    file.close()

    keys = 'data model scalars'.split()
    oldnamefiles = {}
    for key in keys:
        filename = f'id_old_{key}.txt'
        file = open(filename,'w')
        oldnamefiles[key] = file
    
    for line in lines:
        line = line.split()
        for key in keys:
            file = oldnamefiles[key]
            _,prms_id = options(line,key = key)
            file.writelines(prms_id+'\n')
def read_lines(filename):
    file = open(filename,'r')
    oldnames = file.readlines()
    oldnames = [on.strip() for on in oldnames ]
    file.close()
    return oldnames
def change_folder_name():
    keys = 'model model scalars'.split()
    oldnamefiles = {}
    for key in keys:
        filename = f'id_old_{key}.txt'
        oldnames = read_lines(filename)

        filename = f'id_new_{key}.txt'
        newnames =read_lines(filename)

        oldnamefiles[key] = (oldnames,newnames)
    roots = 'evals models scalars'.split()
    for root,key in zip(roots,keys):
        folders = os.listdir('saves/'+root)
        oldnames,newnames = oldnamefiles[key]
        exc_oldnames = [nn for nn in oldnames if nn not in newnames]
        count = 0
        for folder in folders:
            f0 = folder.split('.')[0]
            if f0 not in newnames:
                continue
            assert f0 not in exc_oldnames
            # count += 1
            # i = oldnames.index(f0)
            # f1 = newnames[i]
            # if '.' in folder:
            #     f1 += '.'+ folder.split('.')[1]
            # f0 = os.path.join('saves/'+root,folder)
            # f1 = os.path.join('saves/'+root,f1)
            # print(f0,f1,'\n')
            # # assert os.path.exists(f0)
            # os.rename(f0,f1)

def basic_test():
    st = '--mode data --filtering gaussian'
    line = st.split()
    _,prms_id = options(line,key = 'model')
    print(prms_id)


if __name__ == '__main__':
    basic_test()