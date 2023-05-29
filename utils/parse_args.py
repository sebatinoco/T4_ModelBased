import argparse
from distutils.util import strtobool

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env_name', type = str, default = 'Pendulum-v1', help = 'environment to run the experiment') 
    parser.add_argument('--random', type=lambda x: bool(strtobool(x)), default = False, help = 'boolean to sample random actions if wanted')
    parser.add_argument('--planning_horizon', type = int, default = 30, help = 'planning horizon to generate plan')
    parser.add_argument('--nb_trajectories', type = int, default = 100, help = 'number of trajectories to consider at planning')
    parser.add_argument('--buffer_size', type = int, default = 30000, help = 'replay buffer size')
    parser.add_argument('--nb_data_collection_steps', type = int, default = 10000, help = 'steps to collect data')
    parser.add_argument('--nb_training_steps', type = int, default = 30000, help = 'number of training steps')
    parser.add_argument('--exp_name', type = str, default = 'experiment', help = 'name of the experiment to store results')
    parser.add_argument('--correction', type=lambda x: bool(strtobool(x)), default = True, help = 'boolean to apply correction to pendulum rewards')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args