from Env import Env
import matplotlib.pyplot as plt
import datetime
#import gym
import numpy as np
import itertools
import torch
from SAC.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from SAC.replay_memory import ReplayMemory

args = {'policy':'Gaussian',
        'eval':True,
        'gamma':0.99,
        'tau':0.005,
        'lr':0.0003,
        'alpha':0.2,
        'automatic_entropy_tuning':False,
        'seed':123456,
        'batch_size':256,
        'num_steps':1000001,
        'hidden_size':128,
        'updates_per_step':1,
        'start_steps':10000,
        'target_update_interval':1,
        'replay_size':1000000,
        'cuda':False,
        'observation_size':4,
        'action_size':5,
        'action_space':{'high':np.array([5,1,1,15,15]),'low':np.array([-5,0,0,5,5])},
        '_max_episode_steps':10000
        }

env = Env()

torch.manual_seed(args['seed'])
np.random.seed(args['seed'])
agent = SAC(args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                             args['policy'], "autotune" if args['automatic_entropy_tuning'] else ""))

# Memory
memory = ReplayMemory(args['replay_size'], args['seed'])

# Training Loop
total_numsteps = 0
updates = 0

fig,ax = plt.subplots(figsize=(8,8))
ax.set_xlim([-2,2])
ax.set_ylim([0,4])
# while 1:
#     if env.ball.miss():
#         env.reset()
#     action = [0,1,0,15,0]
#     env.step(action)

#     head_body,ball = env.drawing(ax)
#     plt.pause(0.01)
#     head_body[0].remove()
#     head_body[1].remove()
#     ball.remove()

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    state_xy = []
    r = []

    while not done:
        if args['start_steps'] > total_numsteps:
            action = np.array([np.random.rand()*10-5,np.random.rand(),np.random.rand(),np.random.rand()*25+5,np.random.rand()*25+5])  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args['batch_size']:
            # Number of updates per step in environment
            for i in range(args['updates_per_step']):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args['batch_size'], updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        # action[0] = action[0]*0.7 + (env.ball.x - env.player.x)*0.3
        next_state, reward ,kicked = env.step(action) # Step
        # if not kicked: #kickの方策でkickできなかった場合のreward
        #     reward -= 5
        if env.ball.miss(): #liftingに失敗した時のreward
            reward -= 100
            next_state = env.reset()
            # done = True
        reward += -1 #if time step continues ,add reward
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        #print('state:{},reward:{}'.format(state,reward))

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        #done = True if episode_steps == args['_max_episode_steps'] else False
        if episode_steps == args['_max_episode_steps']/10:
            done = True 

        memory.push(state, action, reward, next_state, done) # Append transition to memory

        state = next_state

        head_body,ball = env.drawing(ax)
        plt.pause(0.001)
        head_body[0].remove()
        head_body[1].remove()
        ball.remove()
        

    if total_numsteps > args['num_steps']:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    '''
    print(np.array(state_x).shape)
    print(np.array(reward).shape)
    print(r)
    '''

    if i_episode % 100 == 0 and args['eval'] is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
