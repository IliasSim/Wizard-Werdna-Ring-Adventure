import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.cuda
import torch.nn.functional as F
# !pip install jupyterplot
# from jupyterplot import ProgressPlot
import torch.nn as nn
import copy
import time
from itertools import count
import math

class ADRQN(nn.Module):
    def __init__(self, n_actions, state_size, embedding_size):
        super(ADRQN, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.embedder = nn.Linear(n_actions, embedding_size)
        self.obs_layer = nn.Linear(state_size, 16)
        self.obs_layer2 = nn.Linear(16,32)
        self.lstm = nn.LSTM(input_size = 32+embedding_size, hidden_size = 128, batch_first = True)
        self.out_layer = nn.Linear(128, n_actions)
    
    def forward(self, observation, action, hidden = None):
        #Takes observations with shape (batch_size, seq_len, obs_dim)
        #Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        action_embedded = self.embedder(action)
        # print(observation.shape)
        observation = F.relu(self.obs_layer(observation))
        observation = F.relu(self.obs_layer2(observation))
        # print(observation.shape)
        lstm_input = torch.cat([observation, action_embedded], dim = -1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)
            
        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out
    
    def act(self, observation, last_action, epsilon, hidden = None):
        q_values, hidden_out = self.forward(observation, last_action, hidden)
        if np.random.uniform() > epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.n_actions)
        return action, hidden_out

class ExpBuffer():
    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, aoarod):
        if self.counter < self.max_storage-1:
            self.counter +=1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = aoarod
    
    def sample(self, batch_size):
        #Returns sizes of (batch_size, seq_len, *) depending on action/observation/return/done
        seq_len = self.sample_length
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            if self.filled - seq_len < 0 :
                raise Exception("Reduce seq_len or increase exploration at start.")
            start_idx = np.random.randint(self.filled-seq_len)
            #print(self.filled)
            #print(start_idx)
            last_act, last_obs, act, rew, obs, done = zip(*self.storage[start_idx:start_idx+seq_len])
            last_actions.append(list(last_act))
            last_observations.append(last_obs)
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append(list(obs))
            dones.append(list(done))
        return torch.tensor(last_actions).cuda(), torch.tensor(last_observations, dtype = torch.float32).cuda(), torch.tensor(actions).cuda(), torch.tensor(rewards).float().cuda() , torch.tensor(observations, dtype = torch.float32).cuda(), torch.tensor(dones).cuda()

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
# print(env.observation_space.shape[0])
n_actions = env.action_space.n
embedding_size = 8
M_episodes = 310
replay_buffer_size = 100000
sample_length = 20
replay_buffer = ExpBuffer(replay_buffer_size, sample_length)
batch_size = 64
eps_start = 0.9
eps = eps_start
eps_end = 0.05
eps_decay = 10
gamma = 0.999
learning_rate = 0.01
blind_prob = 0
EXPLORE = 300

# pp = ProgressPlot(plot_names = ['Return', 'Exploration'], line_names = ['Value'])
adrqn = ADRQN(n_actions, state_size, embedding_size).cuda()
adrqn_target = ADRQN(n_actions, state_size, embedding_size).cuda()
adrqn_target.load_state_dict(adrqn.state_dict())

optimizer = torch.optim.Adam(adrqn.parameters(), lr = learning_rate)

for i_episode in range(M_episodes):
    now = time.time()
    done = False
    hidden = None
    last_action = 0
    current_return = 0
    last_observation = env.reset()
    for t in count():
        action, hidden = adrqn.act(torch.tensor(last_observation).float().view(1,1,-1).cuda(), F.one_hot(torch.tensor(last_action), n_actions).view(1,1,-1).float().cuda(), hidden = hidden, epsilon = eps)
        # env.render()
        observation, reward, done, info = env.step(action)
        if np.random.rand() < blind_prob:
            #Induce partial observability
            observation = np.zeros_like(observation)
        reward = np.sign(reward)
        current_return += reward
        replay_buffer.write_tuple((last_action, last_observation, action, reward, observation, done))
        
        last_action = action
        last_observation = observation
    
        #Updating Networks
        if i_episode > EXPLORE:
                eps = eps_end + (eps_start - eps_end) * math.exp((-1*(i_episode-EXPLORE))/eps_decay)
                # print('I train')
                # print(hidden[0].shape,t)
                last_actions, last_observations, actions, rewards, observations, dones = replay_buffer.sample(batch_size)
                # print(last_observations.shape,rewards.shape)
                q_values, _ = adrqn.forward(last_observations, F.one_hot(last_actions, n_actions).float())
                q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
                predicted_q_values, _ = adrqn_target.forward(observations, F.one_hot(actions, n_actions).float())
                target_values = rewards + (gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])
                #Update network parameters
                optimizer.zero_grad()
                # loss = q_values - target_values.detach()
                loss = torch.nn.MSELoss()(q_values , target_values.detach())
                grads = torch.autograd.backward(loss,retain_graph = True)
                # loss.backward()
                # print(adrqn.obs_layer.weight.grad)
                # optimizer.step()      
        if done:
            break

    # pp.update([[current_return],[eps]])
    adrqn_target.load_state_dict(adrqn.state_dict())

env.close()