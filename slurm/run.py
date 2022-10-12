import argparse
from datetime import date
from run.train import net_train

def options(string_input=[]):
    parser=argparse.ArgumentParser()
    parser.add_argument("-b","--batch",type=int,default=4)
    parser.add_argument("-e","--epoch",type=int,default=4)
    parser.add_argument("-r","--rootdir",type=str,default='/scratch/cg3306/climate/runs/')
    parser.add_argument("--nworkers",type=int,default=1)
    parser.add_argument("-o","--outdir",type=str,default="")
    parser.add_argument("--testrun",type=int,default=0)
    parser.add_argument("--action",type=str,default="train")
    parser.add_argument("--model_id",type=str,default="0")
    parser.add_argument("--data_address",type=str,default=\
                        '/scratch/ag7531/mlruns/19/bae994ef5b694fc49981a0ade317bf07/artifacts/forcing/')
    parser.add_argument("--relog",type=int,default=0)
    parser.add_argument("--rerun",type=int,default=0)
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument("--model_bank_id",type=str,default="0")
    parser.add_argument("--physical_dom_id",type=int,default=0)
    parser.add_argument("--subtime",type=float,default=1)
    parser.add_argument("--disp",type=int,default=500)
    parser.add_argument("--co2",type=int,default=0)
    parser.add_argument("--depth",type=int,default=0)
    if len(string_input)==0:
        return parser.parse_args()
    return parser.parse_args(string_input)


def main():
    today = date.today()
    print("Today's date:", today,flush=True)
    args=options()
    print(args)
    if args.action=="train":
        net_train(args)
    if args.action=="analysis":
        analysis(args)
    if args.action=="binned-r2":
        binned_r2_analysis(args)
    if args.action=="quadratic":
        quadratic_model_matrix(args)
    if args.action=="error-analysis":
        error_analysis(args)
    if args.action=="grad-analysis":
        grad_analysis(args)
    if args.action=="data-cov-analysis":
        data_covariance(args)
    if args.action=="fourier-analysis":
        data_fourier_analysis(args)
    if args.action=="shift-geo-analysis":
        shift_geo_analysis(args)
    if args.action=="grad-probe":
        grad_probe(args)
    if args.action=="prep":
        prep(args)
    if args.action=="projection-analysis":
        projection_analysis(args)
    if args.action=="global-averages":
        global_averages()
    
if __name__=='__main__':
    main()