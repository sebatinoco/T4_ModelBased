import argparse

def run_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--gpu', default = 0, type = int, help = 'gpu to run the experiments')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args