from utils.arguments import populate_data_options
import sys
def main():
    args = sys.argv[1:]
    pargs = populate_data_options(args,parts = (1,1))
    print('\n\n'.join(pargs))

if __name__=='__main__':
    main()
