import os
import yaml
import gym
from train_agent import cartpole_reward, pendulum_reward, train_agent
from mbrl import MBRLAgent
import torch

from utils.run_args import run_args

if __name__ == '__main__':
    
    r_args = run_args()
    
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    # list configs
    configs = sorted(os.listdir('configs'))

    for config in configs:

        # load config
        with open(f"configs/{config}", 'r') as file:
            args = yaml.safe_load(file)

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
                            planning_horizon=30,
                            nb_trajectories=100, 
                            reward_function=reward_function,
                            action_space = action_space,
                            device = device
                            )

        train_agent(env=env, 
                    eval_env=eval_env,
                    agent=mbrl_agent,
                    nb_data_collection_steps=10000,
                    nb_steps_between_model_updates=1000,
                    nb_epochs_for_model_training=10,
                    nb_training_steps=30000,
                    exp_name = env_name,
                    )