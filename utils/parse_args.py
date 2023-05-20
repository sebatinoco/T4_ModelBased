import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env_name', type = str, default = 'Pendulum-v1', help = 'environment to run the experiment')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args