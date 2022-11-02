from data.load import get_data

def main():
    for sigma in [4,8,12,16]:
        args = f'--mode data --sigma {sigma}'.split()
        ds, = get_data(args,torch_flag = False,data_loaders = False)
        ds.save_projections()
        # sdf = ds[0]

   

if __name__=='__main__':
    main()
