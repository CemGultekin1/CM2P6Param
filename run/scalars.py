import sys
from data.load import get_data
from data.save import save_scalar
import numpy as np
from params import SCALAR_PARAMS

normalizations = SCALAR_PARAMS["normalization"]["choices"]

def main():
    args = sys.argv[1:]
    generator,= get_data(args,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('train',))
    tot_scalars = {norm : {} for norm in normalizations}
    time_limit = 64
    for fields,forcings,masks,locations in generator:
        print('time #\t',locations['itime'].numpy()[0])
        if np.any(locations['itime'].numpy() >= time_limit):
            break
        fields = dict(fields,**forcings)
        for name,vec_dict in fields.items():
            vec = vec_dict['val']
            mask = masks[f"{name}_mask"]["val"]
            var = vec[mask>0].numpy()
            mom0 = len(var)
            for norm,scalars in tot_scalars.items():
                if norm == "standard":
                    mom1 = np.sum(var)
                    mom2 = np.sum(var**2)
                else:
                    mom1 = 0
                    mom2 = np.sum(np.abs(var))
                sckey = name
                if sckey not in scalars:
                    scalars[sckey] = [0.,0.,0.]
                scalars[sckey][0] += mom0
                scalars[sckey][1] += mom1
                scalars[sckey][2] += mom2
    for norm,scalars in tot_scalars.items():
        for key,val in scalars.items():
            scalars[key] = [val[1]/val[0],val[2]/val[0]]
        for key,val in scalars.items():
            if norm == "standard":
                scalars[key][1] = np.sqrt(val[1]-val[0]**2)

    def set_normalization(args,name:str):
        if "--normalization" in args:
            i = args.index("--normalization") + 1
            args[i] = name
        else:
            args.append("--normalization")
            args.append(name)
        return args
    for norm,scalars in tot_scalars.items():
        args = set_normalization(args, norm)
        print(f"{norm}:")
        print("\n".join([f"\t\t{name}:\t\t{scs}" for name,scs in scalars.items()]))
        save_scalar(args,scalars)

if __name__=='__main__':
    main()
