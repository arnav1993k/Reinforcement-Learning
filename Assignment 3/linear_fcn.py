import gym
import numpy as np
from collections import deque
from agent import Agent_Linear
import argparse
from utils import *
parser = argparse.ArgumentParser(description='Parameters for Lunar lander')
parser.add_argument('--operation', type = str, default='Train', help='Train or test')
parser.add_argument('--model', type = str, default='Linear')
parser.add_argument('--episodes', type = int, default = 1000, help ='Number of episodes to run')
parser.add_argument('--epsilon', type = float, default = 1, help ='Epsilon for epsilon greedy')
parser.add_argument('--epsilon_decay', type = float, default = 0.99, help ='Epsilon decay rate for epsilon greedy')
parser.add_argument('--discount_factor', type = float, default = 0.99, help ='Discount in value function')
parser.add_argument('--save_path',type = str, default = "./models/", help = "Save directory of for models")
parser.add_argument('--reset',type = bool, default=False, help = "Reset previous models")
parser.add_argument('--batchsize',type = int, default=50, help = "Mini batch size for training")
args = parser.parse_args()
model_type = args.model
operation = args.operation
episodes = args.episodes
epsilon = args.epsilon
decay = args.epsilon_decay
discount_factor = args.discount_factor
save_path = args.save_path
reset = args.reset
batch_size = args.batchsize
print("*** Starting to {} {} model with parameters epsilon = {}, decay = {}, discount_factor = {} ***".format(operation,model_type,epsilon,decay,discount_factor))

env = gym.make('LunarLander-v2')
arr = [2,3,4,5,6,7]
nS = len(arr)
nA = env.action_space.n

agent = Agent_Linear(nS, nA, epsilon, decay , discount_factor)

test = operation

if operation=="test" or operation=="Test":
    agent.model.load_weights(save_path+'trained_agent_lin.h5')
    agent.epsilon = 0
    test = True
else:
    test = False

# Cumulative reward
#arr = [1,2,3,4,5,6,7]
reward_avg = deque(maxlen=100)
rewards = []
avg_loss = []
for i in range(episodes):
    episode_reward = 0
    curr_state = env.reset()
    curr_state = np.reshape(curr_state[arr], [1, nS])

    for time in range(1000):
        if test:
            env.render()
        a,q = agent.action(curr_state)
        
        next_state, reward, done, info = env.step(a)
        episode_reward += reward
        next_state = np.reshape(next_state[arr], [1, nS])
        
        q_next = agent.model.predict(next_state)
        td_target = reward+discount_factor * np.max(q_next)
        q[0][a] = td_target
        agent.train(curr_state,q)
        curr_state = next_state

        if done:
            break

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    # Running average of past 100 episodes
    reward_avg.append(episode_reward)
    rewards.append(np.average(reward_avg))
    avg_loss.append(np.average(agent.loss))
    print ('episode: ', i, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % np.average(reward_avg), ' frames: ', time, ' epsilon: ', '%.2f' % agent.epsilon, ' avg_loss', '%.2f'%np.average(agent.loss))
    
    # with open('trained_agent.txt', 'a') as f:
    #     f.write(str(np.average(reward_avg)) + '\n')

env.close()
if not test:
    agent.model.save(save_path+'trained_agent_lin.h5')
    plot(range(len(rewards)),rewards,"Average reward w.r.t time","Time","Reward","rewards_lin2")
    plot(range(len(avg_loss)),avg_loss,"Loss vs Time","Time","Loss","loss_lin2")