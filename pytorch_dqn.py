import torch.cuda
import torch.nn.functional as F
from torch import nn
from GamePAI_pytorch_dqn_weapon_posses import GamePAI
import sys
import pygame
import numpy
import random
import pandas as pd
from os.path import exists

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
action_number = 8
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
target_network_update = 40000
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



def Agent_Runner(agent,total_episode):
    buffer = Sequence_Buffer()
    dqn_net = ActorCritic().cuda()
    dqn_net_target = ActorCritic().cuda()
    if exists('D:\ekpa\diplomatiki\date_31_5_22_test_4_dqn\gen_model.pt'):
        dqn_net.load_state_dict(torch.load('D:\ekpa\diplomatiki\date_31_5_22_test_4_dqn\gen_model.pt'))
        dqn_net.eval()
        print('Global model is loaded')
    dqn_net_target.load_state_dict(dqn_net.state_dict())
    # local_model.load_state_dict(model.state_dict())
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-2)
    # if agent == 0 or agent == 1:
    # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.0000625, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 2:
    optimizer =  torch.optim.Adam(dqn_net.parameters(), lr=0.00001) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 3:
        # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.00025, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong  
    total_steps = 0
    dfrewards = []
    total_reward = 0
    # torch.manual_seed(0)
    process_cont = True
    g = 0
    while process_cont:
        limit = 'unreached'
        g += 1
        game = GamePAI(1,'Connan',444,444,screenfactor,True,g,False,seeded,agent)
        last_state,last_playerStatus, last_gameText = game.initialGameState()
        done = False
        hiddenCnn_out,hiddenl1_out,hiddenl2_out = None,None,None
        # rewards = []
        total_episode_reward = 0
        steps = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            total_steps += 1
            steps += 1
            q_values,hiddenCnn_out,hiddenl1_out,hiddenl2_out = dqn_net.forward(torch.tensor(last_state).float()[None, None, :].cuda(),torch.tensor(last_playerStatus).float()[None, None, :].cuda(), torch.tensor(last_gameText).float()[None, None, :].cuda(),hiddenCnn_out,hiddenl1_out,hiddenl2_out)            
            if len(buffer.buffer) <= max_explore:
                action = dqn_net.act(q_values,1)
            else:
                e = buffer.CalculateEpsilon(total_steps)
                action = dqn_net.act(q_values,e)
            next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(action)
            if steps > 10000 and game.cave <2:
                limit = 'reached'
                done = True
            buffer.WriteMemory(last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action)
            last_state,last_playerStatus,last_gameText = next_state, next_playerStatus, next_gameText
            if steps%h_step ==0:
                hiddenCnn_out,hiddenl1_out,hiddenl2_out = None,None,None
            total_episode_reward += reward

            if done:
                    print(len(buffer.buffer))
                    total_reward += total_episode_reward

            if len(buffer.buffer)> max_explore:
                if steps%n_step==0 and steps!=0 or done:
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
                            target_q_value,_,_,_ = dqn_net_target.forward(torch.tensor(sequence[i][3]).float()[None, None, :].cuda(),torch.tensor(sequence[i][4]).float()[None, None, :].cuda(), torch.tensor(sequence[i][5]).float()[None, None, :].cuda(),next_h_out,next_hl1_out,next_hl2_out)
                            h_out,hl1_out,hl2_out = next_h_out,next_hl1_out,next_hl2_out
                            # print('Qnetwork ' + str(q_value))
                            # print('target Q net' + str(target_q_value))
                            if pseudo_steps%h_step ==0:
                                h_out,hl1_out,hl2_out = None,None,None
                            if pseudo_steps in episode_list:
                                r = sequence[i][6] + gamma*torch.max(target_q_value)
                                loss = torch.nn.MSELoss()(q_value[0][0][sequence[i][7]] , r.detach())
                                # print(loss)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                if total_steps%target_network_update == 0:
                    print('target_update')
                    dqn_net_target.load_state_dict(dqn_net.state_dict())
                if done:
                    average_reward = total_reward/(g)
                    print("e for agent is {} episode is {} episode reward is {} total reward is {} and avg reward is {} total steps {}".format(e,g,total_episode_reward,total_reward,average_reward,total_steps)+limit)
                    print(len(buffer.buffer))
                    episodeStat = [e,g,total_episode_reward,total_reward,average_reward,total_steps,limit]
                    dfrewards.append(episodeStat)
                    if g%episode_save_model==0 and g != 0:
                        torch.save(dqn_net.state_dict(), 'D:\ekpa\diplomatiki\date_31_5_22_test_4_dqn\gen_model.pt')
                        d = dfrewards
                        d = list(d)
                        dfrewards_data_frame = pd.DataFrame(d, columns=['epsilon','agent episode','episode rewards', 'total rewards', 'mean rewards','total_steps','limit'])
                        destination = r'D:\ekpa\diplomatiki\date_31_5_22_test_4_dqn\agent.xlsx'
                        dfrewards_data_frame.to_excel(destination)
                if g == total_episode:
                    process_cont = False
                    
                    

        
if __name__ == '__main__':
    Agent_Runner(1,10000)
