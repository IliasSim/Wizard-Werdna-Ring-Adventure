import torch
import torch.nn.functional as F
from torch import nn
import torch.multiprocessing as mp
from GamePAI import GamePAI
import sys
import pygame
import numpy
import copy

featuresCNN1 = 32
CNN1Shape = 3
CNN1Step = 1
featuresCNN2 = 32
CNN2Shape = 5
CNN2Step = 2
featuresCNN3 = 16
CNN3Shape = 10
CNN3Step = 5
hiden_layer1 = 512
nnStatusLinearout = 512
nnTextLinearout = 1024
hiden_layer2 = 256
hiden_layer3 = 256
action_number = 8
h_step = 8
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = False
input = (148,148,3)
record = False
truncate = 32
memory_size = truncate
training_list = []
for i in range(1,int(truncate/h_step)+1):
    for z in range(h_step*i+1-8,h_step*i+1):
        training_list.append(z-1)

    
gamma = 0.99
input1 = (1, 1, 148, 148, 3)
input2 = (1,1,10)
input3 = (1,1,125)
beta_entr = 0.001
torch.manual_seed(0)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.cnn1 = nn.Conv2d(3,featuresCNN1,CNN1Shape,stride=CNN1Step)
        self.cnn2 = nn.Conv2d(featuresCNN1,featuresCNN2,CNN2Shape,stride=CNN2Step)
        self.cnn3 = nn.Conv2d(featuresCNN2,featuresCNN3,CNN3Shape,stride=CNN3Step)
        self.lstmCnn = nn.LSTMCell(2704,hiden_layer1)
        self.nnStatus = nn.Linear(10,nnStatusLinearout)
        self.lstmStatus = nn.LSTMCell(nnStatusLinearout,hiden_layer2)
        self.nnText = nn.Linear(125,nnTextLinearout)
        self.lstmText = nn.LSTMCell(nnTextLinearout,hiden_layer3)
        self.value = nn.Linear(hiden_layer1+hiden_layer2+hiden_layer3,1)
        self.act_prob = nn.Linear(hiden_layer1+hiden_layer2+hiden_layer3,action_number)

    def forward(self,inputCNN,input1,input2,h_Cnn = None,h_l1 = None,h_l2 = None,x_Cnn = None,x_l1 = None,x_l2 = None):
        x = self.cnn1(inputCNN)
        c2_in = F.relu(x)
        c3_in = F.relu(self.cnn2(c2_in))
        c_out = F.relu(self.cnn3(c3_in))
        c_out = c_out.view(-1,16*13*13)
        if h_Cnn is not None:
            h_CNN_out, x_Cnn_out = self.lstmCnn(c_out, (h_Cnn,x_Cnn))
        else:
            h_CNN_out, x_Cnn_out = self.lstmCnn(c_out)
        lstml1_in = F.relu(self.nnStatus(input1))
        if h_l1 is not None:
            h_l1_out, x_l1_out = self.lstmStatus(lstml1_in, (h_l1,x_l1))
        else:
            h_l1_out, x_l1_out = self.lstmStatus(lstml1_in)
        lstml2_in = F.relu(self.nnText(input2))
        if h_l2 is not None:
            h_l2_out, x_l2_out = self.lstmText(lstml2_in, (h_l2,x_l2))
        else:
            h_l2_out, x_l2_out = self.lstmText(lstml2_in)
        v = self.value(torch.cat((h_CNN_out,h_l1_out,h_l2_out),1))
        a = F.softmax(self.act_prob(torch.cat((h_CNN_out,h_l1_out,h_l2_out),1)),dim=1)
        return a, v, h_CNN_out, h_l1_out,h_l2_out ,x_Cnn_out, x_l1_out, x_l2_out
    
class Sequence_Buffer:
    def __init__(self):
        self.buffer_size = truncate
        self.buffer = []

    def eraseMemory(self):
        self.buffer = []

    def WriteMemory(self,reward,value,prop,action):
        if len(self.buffer) >= truncate:
            self.buffer = []
        self.buffer.append([reward,value,prop,action])

    def CalculateDiscountReward(self):
        for i in range(len(self.buffer)):
            if i + n_step < len(self.buffer):
                disc_reward = 0
                power = 0
                for z in range(i,i+n_step):
                    disc_reward = disc_reward + self.buffer[z][0]*gamma**power
                    power += 1
                disc_reward = disc_reward + self.buffer[i+n_step][1]*gamma**4
                self.buffer[i][0] = disc_reward
            else:
                disc_reward = 0
                power = 0
                for z in range(i,len(self.buffer)):
                    disc_reward = disc_reward + self.buffer[z][0]*gamma**power
                    power += 1
                self.buffer[i][0] = disc_reward

        # return prop,total_disc_rewards,total_actions

def Agent_Runner(episodes,global_model,agent):
    buffer = Sequence_Buffer()
    local_model = ActorCritic()
    local_model=copy.deepcopy(global_model)
    # local_model.load_state_dict(model.state_dict())
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-2)
    # if agent == 0 or agent == 1:
    # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.0000625, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 2:
    optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.00001, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 3:
        # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.00025, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong  
    total_reward = 0
    # torch.manual_seed(0)
    for g in range(episodes):
        game = GamePAI(1,'Connan',444,444,screenfactor,True,g,False,seeded,agent)
        state,playerStatus, gameText = game.initialGameState()
        done = False
        h_Cnn,h_l1,h_l2 ,x_Cnn,x_l1,x_l2 = None,None,None,None,None,None
        rewards = []
        total_episode_reward = 0
        steps = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            steps += 1
            a,v,h_Cnn,h_l1,h_l2,x_Cnn,x_l1,x_l2 = local_model.forward(torch.tensor(state).float()[None, :],torch.tensor(playerStatus).float()[None, :], torch.tensor(gameText).float()[None, :],h_Cnn,h_l1,h_l2 ,x_Cnn,x_l1,x_l2)
            dist = torch.distributions.Categorical(a)
            action = dist.sample()
            done = False
            next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(action[0])
            print(agent,a,v,steps,reward,action)
            buffer.WriteMemory(reward,v,a,action)
            state,playerStatus,gameText = next_state, next_playerStatus, next_gameText
            total_episode_reward += reward
            rewards.append(reward)
        
            if steps%h_step ==0:
                h_Cnn,h_l1,h_l2,x_Cnn,x_l1,x_l2 = None,None,None,None,None,None

            if steps%truncate==0 and steps!=0 or done:
                buffer.CalculateDiscountReward()
                optimizer.zero_grad()
                total_losses = 0
                print(training_list)
                for i in range(len(buffer.buffer)):
                    if i in training_list:
                        print(buffer.buffer[i][2])
                        # print(buffer.buffer[i][2])
                        td = torch.sub(buffer.buffer[i][0],buffer.buffer[i][1])
                        td_sqr = torch.pow(td,2)
                        log_loss = torch.log(buffer.buffer[i][2][0][buffer.buffer[i][3]])
                        policy_loss = torch.mul(log_loss,td)
                        entropy_loss = torch.sum(torch.log(buffer.buffer[i][2][0]))
                        entropy_loss = torch.neg(entropy_loss)
                        losses = td_sqr + policy_loss - beta_entr*entropy_loss
                        # total_losses += losses
                        losses.backward(retain_graph=True)
                # total_losses.backward()
                for param_l,param_g in zip(local_model.parameters(),global_model.parameters()):
                    param_g.grad =  param_l.grad
                
                optimizer.step()
                local_model=copy.deepcopy(global_model)



            
            if steps == 500:
                done = True
            if done:
                buffer.eraseMemory()
                game.gameOver_pytorch()
        total_reward += total_episode_reward
        average_reward = total_reward/(g+1)
    print("for agent {} total reward after {} episode is {} and avg reward is {}".format(agent,g+1, total_reward, average_reward))

if __name__ == '__main__':
    num_processes = 1 #mp.cpu_count()
    global_model = ActorCritic()
    # for param in global_model.parameters():
        # param = torch.zeros(param.shape)
        # print(param)
    global_model.share_memory()
    processes = []
    for agent in range(num_processes):
        p = mp.Process(target=Agent_Runner, args=(30,global_model,agent))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()