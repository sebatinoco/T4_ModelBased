import os
import yaml
import gym
from train_agent import cartpole_reward, pendulum_reward, train_agent
from mbrl import MBRLAgent
import torch
import time

from utils.run_args import run_args, plot_experiment

if __name__ == '__main__':
    
    # load run args
    r_args = run_args()
    
    filter_env = r_args['env']
    filter_config = r_args['filter_config']
    n_trials = r_args['n_trials']
    
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    # list configs
    configs = sorted(os.listdir('configs'))

    # filter configs if specified
    if filter_env or filter_config:
        
        env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] # filter by environment
        filtered_configs = [config for config in configs if config in filter_config] # filter by config
        
        final_configs = set(env_configs + filtered_configs) # filtered configs
        configs = [config for config in configs if config in final_configs] # filter configs

    print('Running experiments on the following configs: ', configs)

    for trial in range(1, n_trials + 1):
        start_time = time.time()

        for config in configs:

            # load config
            with open(f"configs/{config}", 'r') as file:
                args = yaml.safe_load(file)
                
            # experiment name
            exp_name = f"{args['env_name'][:-3]}_{args['exp_id']}_{trial}"

            env_name = args['env_name']
            env = gym.make(env_name)
            eval_env = gym.make(env_name)

            dim_states = env.observation_space.shape[0]
            continuous_control = isinstance(env.action_space, gym.spaces.Box)
            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n
            reward_function = cartpole_reward if env_name == 'CartPole-v1' else pendulum_reward
            action_space = env.action_space

            mbrl_agent = MBRLAgent(dim_states=dim_states, 
                                dim_actions=dim_actions,
                                continuous_control=continuous_control,
                                model_lr=0.001,
                                buffer_size=30000,
                                batch_size=256,
                                reward_function=reward_function,
                                device = device,
                                **args['agent'],
                                )

            train_agent(env=env, 
                        eval_env=eval_env,
                        agent=mbrl_agent,
                        nb_data_collection_steps=10000,
                        nb_steps_between_model_updates=1000,
                        nb_epochs_for_model_training=10,
                        nb_training_steps=30000,
                        exp_name = exp_name,
                        )
            
        execution_time = time.time() - start_time

        print(f'{execution_time:.2f} seconds -- {(execution_time/60):.2f} minutes -- {(execution_time/3600):.2f} hours')
        

    # plot experiments
    configs = [config.replace('.yaml', '') for config in configs]

    # plot for each experiment
    for config in configs:
        plot_experiment(config)