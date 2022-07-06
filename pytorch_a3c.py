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
hiden_layer1 = 512
nnStatusLinearout = 512
nnTextLinearout = 1024
hiden_layer2 = 256
hiden_layer3 = 256
action_number = 8
minimum_hist = 4
h_step = 8
n_step = 1
screenfactor = 1
decay_steps = 10000
seeded = False
input = (148,148,3)
record = False
truncate = 64
memory_size = truncate
gamma = 0.99
input1 = (1, 1, 37, 37, 5)
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
        self.nnStatus = nn.Linear(8,nnStatusLinearout) # 8 instead of 10
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
        total_states = []
        total_playerstatuses = []
        total_gameTexts = []
        total_disc_rewards = []
        total_actions = []
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
            if len(states) > h_step:
                states = [self.buffer[i][0]]
                playerstatuses = [self.buffer[i][1]]
                gameTexts = [self.buffer[i][2]]
                disc_rewards = [self.buffer[i][3]]
                actions = [self.buffer[i][5]]
            if len(states) > minimum_hist:
                
                # print(disc_rewards)
                total_states.append(numpy.array(states))
                total_playerstatuses.append(numpy.array(playerstatuses))
                total_gameTexts.append(numpy.array(gameTexts))
                total_disc_rewards.append(torch.tensor(disc_rewards).float())
                total_actions.append(torch.tensor(actions).int())
            # check code
        # for i in range(len(total_states)):
            # print(total_states[i].shape)
            # print(torch.tensor(actions))
            # print(torch.tensor(disc_rewards))
        return total_states,total_playerstatuses,total_gameTexts,total_disc_rewards,total_actions






def Agent_Runner(total_episodes,episodes,global_model,dfrewards,agents_reward,lock,agent):
    action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
    buffer = Sequence_Buffer()
    local_model = ActorCritic()
    local_model=copy.deepcopy(global_model)
    # local_model.load_state_dict(model.state_dict())
    # optimizer = torch.optim.RMSprop(global_model.parameters(), lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-2)
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
            buffer.WriteMemory(state,playerStatus,gameText,reward,v.detach(),action)
            state,playerStatus,gameText = next_state, next_playerStatus, next_gameText
            total_episode_reward += reward
            rewards.append(reward)
            # if steps%truncate == 0:
                # print(agent,a,v,steps)
            if steps%truncate==0 and steps!=0 or done:
                print(agent,a)
                states, playerStatuses, gameTextes, disc_rewards,actions = buffer.TakeSequence()
                losses_grad = 0
                # print(len(states),len(buffer.buffer))

                if len(states) !=0:
                    optimizer.zero_grad()
                    for i in range(len(states)):
                        td_total = 0
                        log_total = 0
                        entropy_loss = 0
                        a_t,v_t,_,_,_ = local_model.forward(torch.tensor(states[i]).float()[None, :],torch.tensor(playerStatuses[i]).float()[None, :],torch.tensor(gameTextes[i]).float()[None, :])
                        td = torch.sub(disc_rewards[i],v_t.reshape(1,1,v_t.shape[1]))
                        td_sqr = torch.pow(td,2)
                        td_total = torch.sum(td)
                        td_sqr_total = torch.sum(td_sqr)
                        prob_log = torch.log(a_t)
                        entropy_loss = torch.sum(torch.mul(a_t,prob_log))
                        for z in range(a_t.shape[1]):
                            log_total += torch.log(a_t[0][z][actions[i][z]])
                        entropy_loss = torch.neg(entropy_loss)
                        # entropy_loss = torch.div(entropy_loss,a_t.shape[1])
                        # log_total = torch.div(log_total,a_t.shape[1])
                        policy_losses =  torch.mul(log_total,td_total)
                        losses_grad = td_sqr_total - policy_losses - 0.01*entropy_loss
                        # losses_grad = torch.div(losses_grad,len(states))
                        # print(losses_grad)
                        # print(agent,losses_grad,td_total,log_total,entropy_loss)
                        # optimizer.zero_grad()
                        losses_grad.backward()
                        
                        
                    for param_l,param_g in zip(local_model.parameters(),global_model.parameters()):
                        # print(param_g.grad ==None)
                        param_g.grad =  param_l.grad
                            # print(param_g.grad)
                    # print('I learn')
                    
                    
                optimizer.step()    
                local_model=copy.deepcopy(global_model)
                #for p1, p2 in zip(local_model.parameters(), model.parameters()):
                    #print(torch.equal(p1, p2))
                # local_model.load_state_dict(model.state_dict())
                    # optimizer.zero_grad()
                
                    

                    # print('td'+ str(td))
                    # print()
            if steps%h_step ==0:
                hiddenCnn_out,hiddenl1_out,hiddenl2_out = None,None,None
            
            # if steps == 10:
                # done = True
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
                    torch.save(global_model.state_dict(), 'D:\ekpa\diplomatiki\date_4_7_22_tsak\gen_model.pt')
                    d = dfrewards
                    d = list(d)
                    dfrewards_data_frame = pd.DataFrame(d, columns=['agent','agent episode','episode rewards', 'total agent rewards', 'mean agent rewards','total episodes','total reward','total average reward'])
                    destination = r'D:\ekpa\diplomatiki\date_4_7_22_tsak\agent.xlsx'
                    dfrewards_data_frame.to_excel(destination)
                if episodes[0] == total_episodes + mp.cpu_count():
                    d = dfrewards
                    d = list(d)
                    # print(type(d))
                    dfrewards_data_frame = pd.DataFrame(d, columns=['agent','agent episode','episode rewards', 'total agent rewards', 'mean agent rewards','total episodes','total reward','total average reward'])
                    destination = r'D:\ekpa\diplomatiki\date_4_7_22_tsak\agent.xlsx'
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
    if exists('D:\ekpa\diplomatiki\date_4_7_22_tsak\gen_model.pt'):
        global_model.load_state_dict(torch.load('D:\ekpa\diplomatiki\date_4_7_22_tsak\gen_model.pt'))
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
