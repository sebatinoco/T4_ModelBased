import time
import torch
import datetime
import gym
import csv
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

from mbrl import MBRLAgent
from utils.parse_args import parse_args

def cartpole_reward(observation_batch, action_batch):
    
    # obs = (x, x', theta, theta')
    # rew = cos(theta) - 0.01 * x^2
    
    x = observation_batch[:, 0]
    theta = observation_batch[:, 2]

    return np.cos(theta) - 0.01 * x ** 2

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def pendulum_reward(observation_batch, action_batch, correction = True):
    
    # obs = (cos(theta), sin(theta), theta')
    # rew = - theta^2 - 0.1 * (theta')^2 - 0.001 * a^2
    
    theta_1 = observation_batch[:, 2] # obtain theta'
    action_batch = action_batch.squeeze() # adjust dimensions of action_batch
    cos_theta = observation_batch[:, 0]
    if correction:
        cos_theta = torch.clamp(cos_theta, min = -1, max = 1) # clamp values between -1 and 1   
        theta = torch.arccos(cos_theta) # calculate theta
        
        return - (angle_normalize(theta) ** 2 + 0.1 * theta_1 ** 2 + 0.001 * action_batch ** 2).cpu().numpy()
    
    theta = torch.arccos(cos_theta) # calculate theta
    
    return - (theta ** 2 + 0.1 * theta_1 ** 2 + 0.001 * action_batch ** 2).cpu().numpy()

def train_agent(env, eval_env, agent, nb_training_steps, nb_data_collection_steps,
                nb_epochs_for_model_training, nb_steps_between_model_updates, render=False, exp_name = 'experiment', random = False):

    tr_steps_vec, avg_reward_vec, std_reward_vec = [], [], []
    _, (axes) = plt.subplots(1, 1, figsize=(12,4))

    ob_t = env.reset()
    done = False
    episode_nb = 0
    episode_reward = 0
    episode_steps = 0

    update_performance_metrics(agent, eval_env, 0, axes, tr_steps_vec, 
                               avg_reward_vec, std_reward_vec, random = random)

    print("Collecting data to train the model")
    for model_tr_step in tqdm(range(nb_data_collection_steps)):

        action = agent.select_action(ob_t, random = random)

        ob_t1, reward, done, _ = env.step(action)

        agent.store_transition(ob_t, action, ob_t1)

        ob_t = ob_t1

        if render:
            env.render()
            time.sleep(1 / 60.)

        episode_reward += reward
        episode_steps += 1

        if done:
            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
    
    print("Done collecting data, now training")
    for _ in range(nb_epochs_for_model_training):
        agent.update_model()

    # plot loss
    #agent.plot_loss(exp_name)

    # Part II only
    for tr_step in range(nb_training_steps):

        # update plot
        if (tr_step + 1) % (nb_training_steps / 5) == 0:
            update_performance_metrics(agent, eval_env, tr_step + 1, axes, tr_steps_vec, 
                                        avg_reward_vec, std_reward_vec)

        # update model
        if (tr_step + 1) % nb_steps_between_model_updates == 0:
            print("Updating model")
            for _ in range(nb_epochs_for_model_training):
                agent.update_model()

        action = agent.select_action(ob_t)

        ob_t1, reward, done, _ = env.step(action)

        agent.store_transition(ob_t, action, ob_t1)

        ob_t = ob_t1

        if render:
            env.render()
            time.sleep(1 / 60.)

        episode_reward += reward
        episode_steps += 1

        if done:
            print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4.2f' % \
                        (tr_step + 1, episode_nb + 1, episode_steps, episode_reward))

            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

    plt.savefig(f'figures/experiments/{exp_name}.pdf')
    plt.close()

    save_metrics(agent, tr_steps_vec, avg_reward_vec, std_reward_vec, exp_name)

def update_performance_metrics(agent, eval_env, training_step, axes, tr_steps_vec, avg_reward_vec, std_reward_vec, random = False):

    avg_reward, std_reward = test_agent(eval_env, agent, random = random)

    tr_steps_vec.append(training_step)
    avg_reward_vec.append(avg_reward)
    std_reward_vec.append(std_reward)

    plot_performance_metrics(axes, 
                             tr_steps_vec, 
                             avg_reward_vec, 
                             std_reward_vec)


def save_metrics(agent, tr_steps_vec, avg_reward_vec, std_reward_vec, exp_name):
    with open(f'metrics/{exp_name}.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['steps', 'avg_reward', 'std_reward'])
            for i in range(len(tr_steps_vec)):
                csv_writer.writerow([tr_steps_vec[i], avg_reward_vec[i], std_reward_vec[i]])

             
def test_agent(env, agent, nb_episodes=30, render=False, random = False):

    ep_rewards = []
    avg_steps = 0

    for episode in range(nb_episodes):

        ob_t = env.reset()
        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:

            if render and episode == 0:
                env.render()
                time.sleep(1. / 60)
                
            action = agent.select_action(ob_t, random = random)
            
            ob_t1, reward, done, _ = env.step(action)

            ob_t = ob_t1
            episode_reward += reward
            
            nb_steps += 1

            if done:
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4.2f' % (episode + 1, nb_steps, episode_reward))
    
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    avg_steps /= nb_episodes
    print('Average Reward: %.2f, Reward Deviation: %.2f | Average Steps: %.2f' % (avg_reward, std_reward, avg_steps))

    return avg_reward, std_reward


def plot_performance_metrics(axes, tr_steps_vec, avg_reward_vec, std_reward_vec):
    axes.cla()
    axes.errorbar(tr_steps_vec, avg_reward_vec, yerr=std_reward_vec, marker='.',color='C0')
    axes.set_ylabel('Avg Reward')

    axes.grid('on') 
    axes.set_xlabel('Training step')
    plt.pause(0.05)


if __name__ == '__main__':
    
    args = parse_args()
    
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
                           buffer_size=args['buffer_size'],
                           batch_size=256,
                           planning_horizon=args['planning_horizon'],
                           nb_trajectories=args['nb_trajectories'], 
                           reward_function=reward_function,
                           correction=args['correction'],
                           )

    train_agent(env=env, 
                eval_env=eval_env,
                agent=mbrl_agent,
                nb_data_collection_steps=args['nb_data_collection_steps'],
                nb_steps_between_model_updates=1000,
                nb_epochs_for_model_training=10,
                nb_training_steps=args['nb_training_steps'],
                exp_name = args['exp_name'],
                random = args['random'],
                )