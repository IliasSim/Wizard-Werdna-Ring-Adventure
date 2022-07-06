import torch.cuda
import torch.nn.functional as F
from torch import nn
from GamePAI_pytorch_dqn_weapon_posses import GamePAI
import sys
import pygame
import numpy
import random
import pandas as pd
import os
from os.path import exists
import pickle


featuresCNN1 = 8
CNN1Shape = 8
CNN1Step = 4
featuresCNN2 = 16
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
action_number = 9
minimum_hist = 0
h_step = 8
n_step = 4
screenfactor = 1
seeded = False
input = (148,148,3)
record = False
truncate = 32
memory_size = 350000
max_explore =50000
target_network_update = 1000
epsilon_decay = 4348
episode_save_model = 100
gamma = 0.99
input1 = (1, 1, 148, 148, 3)
input2 = (1,1,10)
input3 = (1,1,125)
episode_list = [5,6,7,13,14,15,21,22,23,29,30,31]
# torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.cnn1 = nn.Conv2d(3,featuresCNN1,CNN1Shape,stride=CNN1Step)
        self.cnn2 = nn.Conv2d(featuresCNN1,featuresCNN2,CNN2Shape,stride=CNN2Step)
        # self.cnn3 = nn.Conv2d(featuresCNN2,featuresCNN3,CNN3Shape,stride=CNN3Step)
        self.lstmCnn = nn.LSTM(4624,hiden_layer1,batch_first=True)
        self.nnStatus = nn.Linear(8,nnStatusLinearout) # 8 instead of 10
        self.lstmStatus = nn.LSTM(nnStatusLinearout,hiden_layer2,batch_first=True)
        self.nnText = nn.Linear(125,nnTextLinearout)
        self.lstmText = nn.LSTM(nnTextLinearout,hiden_layer3,batch_first=True)
        self.Q_value = nn.Linear(hiden_layer1+hiden_layer2+hiden_layer3,action_number)
        

    
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
        if hiddenl2 is not None:
            lstml2_out, hiddenl2_out = self.lstmText(lstml2_in, hiddenl2)
        else:
            lstml2_out, hiddenl2_out = self.lstmText(lstml2_in)
        q_values = self.Q_value(torch.cat((lstmCnn_out,lstml1_out,lstml2_out),2))
        return q_values,hiddenCnn_out,hiddenl1_out,hiddenl2_out

    def act(self,q_values,epsilon):
        if numpy.random.uniform() > epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = numpy.random.randint(action_number)
        return action

class Sequence_Buffer:
    def __init__(self):
        self.buffer_size = truncate
        self.buffer = []
        self.epsilon = 1

    def eraseMemory(self):
        self.buffer = []

    def WriteMemory(self,last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action):
        if len(self.buffer) >= memory_size:
            del self.buffer[0]
        self.buffer.append([last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action])

    def TakeSequence(self,batch_size):
        batch_start_boolean = True
        if len(self.buffer) >= batch_size:
            while batch_start_boolean:
                batch_start = random.randint(0,len(self.buffer)-1)
                if len(self.buffer) - batch_start >= batch_size:
                    batch_start_boolean = False
            sequence = self.buffer[batch_start:(batch_start+batch_size)]   
        return sequence

    def CalculateEpsilon(self,total_steps):
        if total_steps%epsilon_decay == 0 and total_steps !=0:
            self.epsilon = self.epsilon*gamma
        if self.epsilon <= 0.1:
            self.epsilon = 0.1
        return self.epsilon



def Agent_Runner():
    buffer = Sequence_Buffer()
    dqn_net = ActorCritic().cuda()
    if exists('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt'):
        size = os.path.getsize('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt') 
    else: 
        with open('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'x') as f:
            f.write('')
            size = 0
    if size >0:
        with open ('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'rb') as fp:
            buffer.buffer = pickle.load(fp)
    if exists('D:\ekpa\diplomatiki\date_31_5_22_test_5_dqn\gen_model.pt'):
        dqn_net.load_state_dict(torch.load('D:\ekpa\diplomatiki\date_31_5_22_test_5_dqn\gen_model.pt'))
        dqn_net.eval()
        print('Global model is loaded')
    # local_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.RMSprop(dqn_net.parameters(), lr=1e-1, alpha=0.95, momentum=0.95, eps=1e-2)
    # if agent == 0 or agent == 1:
    # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.0000625, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 2:
    # optimizer =  torch.optim.Adam(dqn_net.parameters(), lr=0.00025) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 3:
        # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.00025, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong  
    train_epoch = 23999
    total_loss = 0
    # torch.manual_seed(0)
    training_cont = True
    # dqn_net.train()
    dflosses = []
    action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'attack',7:'pick',8:'mp'}

    while training_cont:
        train_epoch += 1
        print(train_epoch)
        # print('training start')
        sequence = buffer.TakeSequence(truncate)
        pseudo_steps = 0
        loss = 0
        h_out,hl1_out,hl2_out = None,None,None
        for i in range(len(sequence)):
            pseudo_steps += 1
            if h_out is not None:
                h_out = tuple(state.detach() for state in h_out)
                hl1_out = tuple(state.detach() for state in hl1_out)
                hl2_out = tuple(state.detach() for state in hl2_out)
            
            q_value,next_h_out,next_hl1_out,next_hl2_out = dqn_net.forward(torch.tensor(sequence[i][0]).float()[None, None, :].cuda(),torch.tensor(sequence[i][1]).float()[None, None, :].cuda(), torch.tensor(sequence[i][2]).float()[None, None, :].cuda(),h_out,hl1_out,hl2_out)
            target_q_value,_,_,_ = dqn_net.forward(torch.tensor(sequence[i][3]).float()[None, None, :].cuda(),torch.tensor(sequence[i][4]).float()[None, None, :].cuda(), torch.tensor(sequence[i][5]).float()[None, None, :].cuda(),next_h_out,next_hl1_out,next_hl2_out)
            
            h_out,hl1_out,hl2_out = next_h_out,next_hl1_out,next_hl2_out
                            # print('Qnetwork ' + str(q_value))
                            # print('target Q net' + str(target_q_value))
            if pseudo_steps%h_step ==0:
                h_out,hl1_out,hl2_out = None,None,None
            if pseudo_steps in episode_list:
                r = sequence[i][6] + gamma*torch.max(target_q_value)
                loss = torch.nn.MSELoss()(q_value[0][0][sequence[i][7]] , r.detach())
                total_loss += loss
                average_loss = total_loss/train_epoch
                print(loss,action_name[torch.argmax(q_value).item()],action_name[sequence[i][7]],action_name[torch.argmax(q_value).item()]==action_name[sequence[i][7]])                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        average_loss = total_loss/train_epoch
        # print(total_loss)
        # print('average: ' + str(average_loss))
        if train_epoch%target_network_update == 0:
            print(total_loss)
            print('average: ' + str(average_loss))
        if train_epoch%(target_network_update*8) == 0:
            episodeStat = [total_loss.item(),average_loss.item()]
            dflosses.append(episodeStat)
            d = dflosses
            d = list(d)
            dflosses_data_frame = pd.DataFrame(d, columns=['total_losses','averarage_losses'])
            destination = r'D:\ekpa\diplomatiki\date_31_5_22_test_5_dqn\agent.xlsx'
            dflosses_data_frame.to_excel(destination)
            torch.save(dqn_net.state_dict(), 'D:\ekpa\diplomatiki\date_31_5_22_test_5_dqn\gen_model_ndq_'+ str(train_epoch) + '.pt')                   
                    

        
if __name__ == '__main__':
    Agent_Runner()
