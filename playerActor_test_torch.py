import torch
import torch.nn.functional as F
from torch import nn
import torch.multiprocessing as mp
from GamePAI_pytorch_dqn_weapon_posses_tsak import GamePAI
import sys
import pygame
import numpy
import copy
import pandas as pd
from os.path import exists

featuresCNN1 = 16
CNN1Shape = 8
CNN1Step = 4
featuresCNN2 = 32
CNN2Shape = 4
CNN2Step = 2
featuresCNN3 = 64
CNN3Shape = 3
CNN3Step = 1
hiden_layer1 = 512
nnStatusLinearout = 512
nnTextLinearout = 1024
hiden_layer2 = 256
hiden_layer3 = 256
action_number = 8
minimum_hist = 0
h_step = 4
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = True
input = (148,148,3)
record = False
truncate = 32
memory_size = truncate
gamma = 0.99
input1 = (1, 1, 37, 37, 3)
input2 = (1,1,10)
input3 = (1,1,125)
# torch.manual_seed(0)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.cnn1 = nn.Conv2d(5,featuresCNN1,CNN1Shape,stride=CNN1Step)
        self.cnn2 = nn.Conv2d(featuresCNN1,featuresCNN2,CNN2Shape,stride=CNN2Step)
        # self.cnn3 = nn.Conv2d(featuresCNN2,featuresCNN3,CNN3Shape,stride=CNN3Step)
        self.lstmCnn = nn.LSTM(288,hiden_layer1,batch_first=True)
        self.nnStatus = nn.Linear(8,nnStatusLinearout)
        self.lstmStatus = nn.LSTM(nnStatusLinearout,hiden_layer2,batch_first=True)
        self.nnText = nn.Linear(125,nnTextLinearout)
        self.lstmText = nn.LSTM(nnTextLinearout,hiden_layer3,batch_first=True)
        self.value = nn.Linear(hiden_layer1+hiden_layer2+hiden_layer3,1)
        self.act_prob = nn.Linear(hiden_layer1+hiden_layer2+hiden_layer3,action_number)
        

    
    def forward(self,input1,input2,input3,hiddenCnn = None,hiddenl1 = None,hiddenl2 = None):
        batch_size, timesteps, C, H, W = input1.size()
        c1_in = input1.view(batch_size * timesteps, C, H, W)
        # print(c1_in.shape,c1_in.type())
        x = self.cnn1(c1_in)
        c2_in = F.relu(x)
        c3_in = F.relu(self.cnn2(c2_in))
        # c_out = F.relu(self.cnn3(c3_in))
        lstmCnn_in = c3_in.view(batch_size, timesteps, -1)
        # print(lstmCnn_in.shape)
        if hiddenCnn is not None:
            lstmCnn_out, hiddenCnn_out = self.lstmCnn(lstmCnn_in, hiddenCnn)
        else:
            lstmCnn_out, hiddenCnn_out = self.lstmCnn(lstmCnn_in)
        batch_size, timesteps,status  = input2.size()
        l1_in = input2.view(batch_size * timesteps, status)
        l1_in = l1_in.view(batch_size, timesteps, -1)
        lstml1_in = F.relu(self.nnStatus(l1_in))
        if hiddenl1 is not None:
            lstml1_out, hiddenl1_out = self.lstmStatus(lstml1_in, hiddenl1)
        else:
            lstml1_out, hiddenl1_out = self.lstmStatus(lstml1_in)
        batch_size, timesteps,status  = input3.size()
        l2_in = input3.view(batch_size * timesteps, status)
        l2_in = l2_in.view(batch_size, timesteps, -1)
        lstml2_in = F.relu(self.nnText(l2_in))
        # print(hiddenl2 is None)
        if hiddenl2 is not None:
            lstml2_out, hiddenl2_out = self.lstmText(lstml2_in, hiddenl2)
        else:
            lstml2_out, hiddenl2_out = self.lstmText(lstml2_in)
        v = self.value(torch.cat((lstmCnn_out,lstml1_out,lstml2_out),2))
        a = F.softmax(self.act_prob(torch.cat((lstmCnn_out,lstml1_out,lstml2_out),2)),dim=2)
        return a,v,hiddenCnn_out,hiddenl1_out,hiddenl2_out








def Agent_Runner(global_model,agent):
    action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'attack',7:'pick'}
    local_model = ActorCritic()
    local_model=copy.deepcopy(global_model)
    total_reward = 0
    # dfrewards = []
    # torch.manual_seed(0)
    process_cont = True
    g = 0
    while process_cont:
        g += 1
        game = GamePAI(1,'Connan',444,444,screenfactor,True,g,False,seeded,agent)
        state,playerStatus, gameText = game.initialGameState()
        done = False
        hiddenCnn_out,hiddenl1_out,hiddenl2_out = None,None,None
        rewards = []
        total_episode_reward = 0
        steps = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            steps += 1
            a,v,hiddenCnn_out,hiddenl1_out,hiddenl2_out = local_model.forward(torch.tensor(state).float()[None, None, :],torch.tensor(playerStatus).float()[None, None, :], torch.tensor(gameText).float()[None, None, :],hiddenCnn_out,hiddenl1_out,hiddenl2_out)
            dist = torch.distributions.Categorical(a)
            action = dist.sample()
            done = False
            next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(action[0][-1])
            print(agent,action_name[action.item()],a,steps)
            state,playerStatus,gameText = next_state, next_playerStatus, next_gameText
            total_episode_reward += reward
            rewards.append(reward)
            
            # if steps%h_step ==0:
                # hiddenCnn_out,hiddenl1_out,hiddenl2_out = None,None,None
            
            # if steps == 10:
                # done = True
            if done:
                game.gameOver_pytorch()
                total_reward += total_episode_reward
                average_reward = total_reward/g
                total_average_reward = total_reward/g
                print("for agent {} episode is {} episode reward is {} total reward for the agent is {} and avg reward is {} ".format(agent,g,total_episode_reward, total_reward,average_reward))
                print(a)
            

        
if __name__ == '__main__':

    global_model = ActorCritic()
    if exists('D:\ekpa\diplomatiki\date_4_7_22_tsak\gen_model.pt'):
        global_model.load_state_dict(torch.load('D:\ekpa\diplomatiki\date_4_7_22_tsak\gen_model.pt'))
        global_model.eval()
        print('Global model is loaded')
    global_model.share_memory()
    processes = []
    Agent_Runner(global_model,0)
    # for agent in range(num_processes):
        # p = mp.Process(target=Agent_Runner, args=(1000,episodes,global_model,dfrewards,agents_reward,lock,agent))
        # p.start()
        # processes.append(p)
    # for p in processes:
        # p.join()
