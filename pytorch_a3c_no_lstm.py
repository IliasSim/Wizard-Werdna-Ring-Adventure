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
# featuresCNN3 = 64
# CNN3Shape = 3
# CNN3Step = 1
# featuresCNN3 = 64
# CNN3Shape = 3
# CNN3Step = 1
linear_layer_cnn = 512
nnStatusLinearout = 512
nnTextLinearout = 1024
hiden_layer2 = 256
hiden_layer3 = 256
action_number = 8
minimum_hist = 4
h_step = 8
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = False
input = (148,148,3)
record = False
truncate = 64
memory_size = truncate
gamma = 0.99
input1 = (1, 148, 148, 5)
input2 = (1,10)
input3 = (1,125)
# torch.manual_seed(0)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.cnn1 = nn.Conv2d(5,featuresCNN1,CNN1Shape,stride=CNN1Step)
        self.cnn2 = nn.Conv2d(featuresCNN1,featuresCNN2,CNN2Shape,stride=CNN2Step)
        # self.cnn3 = nn.Conv2d(featuresCNN2,featuresCNN3,CNN3Shape,stride=CNN3Step)
        self.linear_Cnn = nn.Linear(288,linear_layer_cnn)
        self.nnStatus = nn.Linear(8,nnStatusLinearout) # 8 instead of 10
        # self.lstmStatus = nn.LSTM(nnStatusLinearout,hiden_layer2,batch_first=True)
        self.nnText = nn.Linear(125,nnTextLinearout)
        # self.lstmText = nn.LSTM(nnTextLinearout,hiden_layer3,batch_first=True)
        self.value = nn.Linear(linear_layer_cnn+nnStatusLinearout+nnTextLinearout,1)
        self.act_prob = nn.Linear(linear_layer_cnn+nnStatusLinearout+nnTextLinearout,action_number)
        

    
    def forward(self,input1,input2,input3):
        # print(c1_in.shape,c1_in.type())
        x = self.cnn1(input1)
        c2_in = F.relu(x)
        c3_in = F.relu(self.cnn2(c2_in))
        c3_flat = torch.flatten(c3_in) 
        cnn_out = F.relu(self.linear_Cnn(c3_flat))
        cnn_out = cnn_out[None,:]
        nnStatus = F.relu(self.nnStatus(input2))
        nnText = F.relu(self.nnText(input3))
        v = self.value(torch.cat((cnn_out,nnStatus,nnText),1))
        a = F.softmax(self.act_prob(torch.cat((cnn_out,nnStatus,nnText),1)),dim=1)
        return a,v

class Sequence_Buffer:
    def __init__(self):
        self.buffer_size = truncate
        self.buffer = []

    def eraseMemory(self):
        self.buffer = []

    def WriteMemory(self,state,playerStatus,gameText,disc_reward,value,action):
        if len(self.buffer) >= truncate:
            self.buffer = []
        self.buffer.append([state,playerStatus,gameText,disc_reward,value,action])

    def TakeSequence(self):
        states = []
        playerstatuses = []
        gameTexts = []
        disc_rewards = []
        actions = []
        for i in range(len(self.buffer)):
            if i + n_step < len(self.buffer):
                disc_reward = 0
                power = 0
                for z in range(i,i+n_step):
                    disc_reward = disc_reward + self.buffer[z][3]*gamma**power
                    power += 1
                disc_reward = disc_reward + self.buffer[i+n_step][4]*gamma**n_step
                self.buffer[i][3] = disc_reward
            else:
                disc_reward = 0
                power = 0
                for z in range(i,len(self.buffer)):
                    disc_reward = disc_reward + self.buffer[z][3]*gamma**power
                    power += 1
                self.buffer[i][3] = disc_reward
            # print(self.buffer[i][3])
        for i in range(len(self.buffer)):
            states.append(self.buffer[i][0])
            playerstatuses.append(self.buffer[i][1])
            gameTexts.append(self.buffer[i][2])
            disc_rewards.append(self.buffer[i][3])
            actions.append(self.buffer[i][5])
            
        return states,playerstatuses,gameTexts,disc_rewards,actions






def Agent_Runner(total_episodes,episodes,global_model,dfrewards,agents_reward,lock,agent):
    action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
    buffer = Sequence_Buffer()
    local_model = ActorCritic()
    local_model=copy.deepcopy(global_model)
    # local_model.load_state_dict(model.state_dict())
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-2)
    # if agent == 0 or agent == 1:
    # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.0000625, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 2:
    optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.00001) # 0.00001 for breakout, 0.00025 is faster for pong
    # if agent == 3:
        # optimizer =  torch.optim.Adam(global_model.parameters(), lr=0.00025, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong  
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
        rewards = []
        total_episode_reward = 0
        steps = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            steps += 1
            a,v = local_model.forward(torch.tensor(state).float()[None, :],torch.tensor(playerStatus).float()[None, :], torch.tensor(gameText).float()[None, :])
            dist = torch.distributions.Categorical(a)
            action = dist.sample()
            done = False
            next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(action[0])
            buffer.WriteMemory(state,playerStatus,gameText,reward,v.detach(),action)
            state,playerStatus,gameText = next_state, next_playerStatus, next_gameText
            total_episode_reward += reward
            rewards.append(reward)
            # if steps%truncate == 0:
                # print(agent,a,v,steps)
            if steps%truncate==0 and steps!=0 or done:
                print(agent,a)
                states, playerStatuses, gameTextes, disc_rewards,actions = buffer.TakeSequence()
                # print(len(states),len(buffer.buffer))
                if len(states) !=0:
                    optimizer.zero_grad()
                    for i in range(len(states)):
                        # print(len(states))
                        entropy_loss = 0
                        a_t,v_t= local_model.forward(torch.tensor(states[i]).float()[None, :],torch.tensor(playerStatuses[i]).float()[None, :],torch.tensor(gameTextes[i]).float()[None, :])
                        td = torch.sub(disc_rewards[i],v_t)
                        td_sqr = torch.pow(td,2)
                        prob_log = torch.log(a_t)
                        entropy_loss = torch.sum(torch.mul(a_t,prob_log))
                        policy_losses = prob_log[0][actions[i]]
                        entropy_loss = torch.neg(entropy_loss)
                        losses_grad = td_sqr - policy_losses - 0.01*entropy_loss
                        losses_grad.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    for param_l,param_g in zip(local_model.parameters(),global_model.parameters()):
                        param_g.grad =  param_l.grad
                            # print(param_g.grad)
                    # print('I learn')
                    
                    
                    # optimizer.step()
                local_model=copy.deepcopy(global_model)
             
            if done:
                lock.acquire()
                agents_reward[0] += total_episode_reward
                episodes[0] += 1
                buffer.eraseMemory()
                game.gameOver_pytorch()
                total_reward += total_episode_reward
                average_reward = total_reward/(g)
                total_average_reward = agents_reward[0]/episodes[0]
                print("for agent {} episode is {} episode reward is {} total reward for the agent is {} and avg reward is {} total episode number is {} all process reward is {} and total average is {}".format(agent,g,total_episode_reward, total_reward,average_reward,episodes[0],agents_reward[0],total_average_reward))
                print(a)
                episodeStat = [agent,g,total_episode_reward, total_reward,average_reward,episodes[0],agents_reward[0],total_average_reward]
                dfrewards.append(episodeStat)
                lock.release()
                if episodes[0] > total_episodes:
                    process_cont = False
                if episodes[0]%100==0 and episodes[0] != 0:
                    torch.save(global_model.state_dict(), 'D:\ekpa\diplomatiki\date_4_7_22_tsak_simple\gen_model.pt')
                    d = dfrewards
                    d = list(d)
                    dfrewards_data_frame = pd.DataFrame(d, columns=['agent','agent episode','episode rewards', 'total agent rewards', 'mean agent rewards','total episodes','total reward','total average reward'])
                    destination = r'D:\ekpa\diplomatiki\date_4_7_22_tsak_simple\agent.xlsx'
                    dfrewards_data_frame.to_excel(destination)
                if episodes[0] == total_episodes + mp.cpu_count():
                    d = dfrewards
                    d = list(d)
                    # print(type(d))
                    dfrewards_data_frame = pd.DataFrame(d, columns=['agent','agent episode','episode rewards', 'total agent rewards', 'mean agent rewards','total episodes','total reward','total average reward'])
                    destination = r'D:\ekpa\diplomatiki\date_4_7_22_tsak_simple\agent.xlsx'
                    dfrewards_data_frame.to_excel(destination)
                    process_cont = False

    
    print("for agent {} total reward after {} episode is {} and avg reward is {}".format(agent,g, total_reward, average_reward))
            

        
if __name__ == '__main__':
    num_processes = mp.cpu_count()
    episodes = mp.Manager().list([0])
    dfrewards = mp.Manager().list()
    agents_reward = mp.Manager().list([0])
    lock = mp.Lock()

    global_model = ActorCritic()
    if exists('D:\ekpa\diplomatiki\date_4_7_22_tsak_simple\gen_model.pt'):
        global_model.load_state_dict(torch.load('D:\ekpa\diplomatiki\date_4_7_22_tsak_simple\gen_model.pt'))
        global_model.eval()
        print('Global model is loaded')
    global_model.share_memory()
    processes = []
    for agent in range(num_processes):
        p = mp.Process(target=Agent_Runner, args=(10000,episodes,global_model,dfrewards,agents_reward,lock,agent))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()