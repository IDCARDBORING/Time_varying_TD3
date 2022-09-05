import mujoco_py
import gym
import time
import numpy as np
from agent import TD3
import matplotlib.pyplot as plt
import torch
import time
import random
import math
from utils import save_results,make_dir
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
curr_path = os.path.dirname(__file__)
torch.cuda.set_device(2)
class TD3Config:
    def __init__(self) -> None:
        self.algo = 'TD3'
        self.env = 'Plane-v0'
        self.seed = 0
        self.result_path = curr_path + "/results/" + self.env + '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/results/" + self.env + '/' + curr_time + '/models/'  # path to save models
        self.start_timestep = 50  # Time steps initial random policy is used
        self.eval_freq = 5e3  # How often (time steps) we evaluate
        # self.train_eps = 800
        self.max_epi = 2000000  # epi
        self.max_step = 2000
        self.expl_noise = 0.1  # Std of Gaussian exploration noise
        self.batch_size = 256  # Batch size for both actor and critic 256
        self.gamma = 0.99  # gamma factor
        self.lr = 0.0002  # Target network update rate
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def eval(agent,enva_env,eval_episodes):
    avg_reward=0
    for _ in range(eval_episodes):
        state, done = enva_env.reset(), False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, _ = enva_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("eval done:"+str(avg_reward))
    return avg_reward
def main(seed):
    cfg = TD3Config()
    random_seed = seed
    # ant:3000/5000  5628
    # hopper 2500-3500
    # humanoid 4000-5000  0.05
    # inver 10000

    envlists = ['Walker2d-v2',
                'HalfCheetah-v2',
                'Ant-v2',
                'Hopper-v2',
                'Humanoid-v2',
                'InvertedDoublePendulum-v2']
    env_sign = int(input(
        'choosing Env! Walker2DMuJoCoEnv-v0, HalfCheetahMuJoCoEnv-v0, AntMuJoCoEnv-v0,'
        ' HopperMuJoCoEnv-v0, HumanoidMuJoCoEnv-v0, InvertedDoublePendulumMuJoCoEnv-v0'))
    print('selected RL environment is:' + envlists[env_sign])
    env = gym.make(envlists[env_sign])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(state_dim)
    print(action_dim)
    print(max_action)
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    model = TD3(state_dim,action_dim,max_action,cfg)
    all_ep_r = []
    total_step=0
    MSE_REWARDS = []
    reward_max = 0
    reward_min = 0
    eval_time=0
    t=0
    for i_episode in range(cfg.max_epi):
        ep_r = 0
        rewards = []
        state = env.reset().astype(np.float32)
        for t in range(2000):
            total_step+=1
            if total_step % 5000 == 0:
                eval_reward = eval(enva_env=env, eval_episodes=3, agent=model)
                eval_time+=1
                writer.add_scalar(tag=envlists[env_sign] + str(random_seed)+"/reward_normal", scalar_value=eval_reward,
                                  global_step=eval_time)
                break
            action = (model.select_action(state) + np.random.normal(0, max_action * cfg.expl_noise, size=action_dim)
                     ).clip(-max_action, max_action)
            s_prime, reward, done, info = env.step(action)
            done_bool = float(done)
            # Store data in replay buffer
            model.memory.push(state, action, s_prime, reward, done_bool)
            rewards.append(reward)
            if i_episode > 50:
                model.update()
            if done:
                break
            state = s_prime.astype(np.float32)
            ep_r += reward

        ## mine
        # if ep_r>reward_max:
        #     reward_max=ep_r
        # elif ep_r<reward_min:
        #     reward_min=ep_r
        # if i_episode > 50:
        #     MSE_REWARDS.pop(0)
        #     MSE_REWARDS.append(ep_r)
        #     cfg.expl_noise = 0.05 * np.exp(-math.sqrt(i_episode) * np.math.log(np.e, 10) / 15) + 0.05 * np.exp(
        #         -(np.sum(MSE_REWARDS) - len(MSE_REWARDS) * reward_min) / (reward_max - reward_min)) + 0.06
        # else:
        #     MSE_REWARDS.append(ep_r)
        #
        # model.policy_noise=cfg.expl_noise
        # writer.add_scalar(tag=envlists[env_sign] + str(random_seed)+"/exp_noise", scalar_value=cfg.expl_noise,
        #                   global_step=eval_time)
        if i_episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print('seed:', random_seed, 'episode:', i_episode, 'score:', ep_r, 'step:', t, 'max:', max(all_ep_r))

    # make_dir(cfg.result_path, cfg.model_path)
    # model.save(path=cfg.model_path)
    env.close()

if __name__ == '__main__':
    load_path = os.path.split(os.path.realpath(__file__))[0]
    logdir = load_path + '/log/traintest'
    writer = SummaryWriter(log_dir=logdir)
    main(random.randint(0,1000))