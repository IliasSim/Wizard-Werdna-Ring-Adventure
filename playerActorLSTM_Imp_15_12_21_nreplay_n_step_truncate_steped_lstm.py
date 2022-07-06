from GamePAI import GamePAI
import pygame
import numpy as np
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import gc
import tensorflow.keras.losses as kls
from os.path import exists
import sys
import random
# from guppy import hpy


featuresCNN1 = 32
CNN1Shape = 3
CNN1Step = 1
featuresCNN2 = 32
CNN2Shape = 5
CNN2Step = 2
# maxpooling
featuresCNN3 = 16
CNN3Shape = 10
CNN3Step = 5
denseLayerN = 256
denseLayerNL_2 = 32
denseLayerNL_3= 64
denseLayerNL_21 = 128
denseLayerNL_31 = 256 
h_step = 8
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = False
input = (148,148,3)
record = False
memory_size = n_step + h_step - 1
batch_size = 12
truncate = 100
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
class critic(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv3D(featuresCNN1,(1,CNN1Shape,CNN1Shape),(1,CNN1Step,CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv3D(featuresCNN2,(1,CNN2Shape,CNN2Shape),(1,CNN2Step,CNN2Step),activation = 'relu')
        self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((-1,2704))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        self.v = tf.keras.layers.Dense(1, activation =None)
      

    def call(self,input_data,input_data1,input_data2):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        y = self.l7(input_data1)
        z = self.l9(input_data2)
        h = self.conc1((y,z))
        g = self.conc2((h,x))
        v = self.v(g)
        return v

class actor(tf.keras.Model):
    '''This class creates the model of the actor part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv3D(featuresCNN1,(1,CNN1Shape,CNN1Shape),(1,CNN1Step,CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv3D(featuresCNN2,(1,CNN2Shape,CNN2Shape),(1,CNN2Step,CNN2Step),activation = 'relu')
        self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((-1,2704))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        self.a = tf.keras.layers.Dense(9, activation ='softmax')
      

    def call(self,input_data,input_data1,input_data2):
        print(input_data.shape,input_data1.shape,input_data2.shape)
        x = self.l1(input_data)
        x = self.l2(x)
        x= self.l3(x)
        x = self.l4(x)
        x= self.l5(x)
        y = self.l7(input_data1)
        z = self.l9(input_data2)
        h = self.conc1((y,z))
        g = self.conc2((h,x))
        a = self.a(g)
        return a

class agent():
    '''agent is a class that creates the agent who play the game using the actor and critic network.
    Also contains functions for the training of the networks'''
    def __init__(self):
        # self.gamma = gamma
        # self.initial_learning_rate = initial_learning_rate
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate=self.initial_learning_rate,
        # decay_steps=decay_steps,
        # decay_rate=0.9)
        self.a_opt = tf.keras.optimizers.Adam(1e-5)
        self.c_opt = tf.keras.optimizers.Adam(1e-5)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None
        self.buffer_State = []
        self.buffer_playerStatus = []
        self.buffer_text = []
        self.buffer_reward = []
        self.buffer_size = h_step

    def act(self,state,playerstatus,gameText):
        '''This function use the actor NN in order to produce an action as an output'''
        prob = self.actor(state,playerstatus,gameText)
        print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def preprocess0(self, state,playerstatus,gameText):
        '''This function cretaes the history of observation for the input to the NN. Also calculate
        the valeu of the last observation based on past and current rewards.'''
       
        if len(self.buffer_State) >= h_step:
            self.buffer_State = []
        self.buffer_State.append(state)
        state = np.array(self.buffer_State,dtype=np.float32)
        state = np.expand_dims(state, axis=0)

        if len(self.buffer_playerStatus) >= h_step:
            self.buffer_playerStatus = []
        self.buffer_playerStatus.append(playerstatus)
        playerstatus = np.array(self.buffer_playerStatus, dtype=np.float32)
        playerstatus = np.expand_dims(playerstatus, axis=0)

        if len(self.buffer_text) >= h_step:
            self.buffer_text = []
        self.buffer_text.append(gameText)
        gameText = np.array(self.buffer_text, dtype=np.float32)
        gameText = np.expand_dims(gameText, axis=0)
        # print(state.shape)
        return state,playerstatus,gameText

    def actor_loss(self, probs, actions, td):
        '''This function calculate actor NN losses. Which is negative of Log probability of action taken multiplied 
        by temporal difference used in q learning.'''
        # probability = []
        # log_probability= []
        dist = tfp.distributions.Categorical(probs, dtype=tf.float32)
        log_prob = dist.log_prob(actions)
        prob = dist.prob(actions)
        # probability.append(prob)
        # log_probability.append(log_prob)
        # p_loss= []
        # e_loss = []
        td = td.numpy()
        
        td =  tf.constant(td)
        policy_loss = tf.math.multiply(log_prob,td)
        entropy_loss = tf.math.negative(tf.math.multiply(prob,log_prob))
        # p_loss.append(policy_loss)
        # e_loss.append(entropy_loss)
        # print(p_loss)
        # p_loss = tf.stack(p_loss)
        # e_loss = tf.stack(e_loss)
        # p_loss = tf.reduce_mean(p_loss)
        # e_loss = tf.reduce_mean(e_loss)
        loss = -policy_loss - 0.0001 * entropy_loss
        return loss

    def learn(self, states,playerstatus,gameTexts, G,actions):
        '''This function is used for the training of the network, it also contain a code chunk for the depiction of the image used for the training. 
        For critic loss, we took a naive way by just taking the square of the temporal difference.'''
        # discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states,playerstatus,gameTexts, training=True)
            v =  self.critic(states,playerstatus,gameTexts,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(G, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(G, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        
        if record:
            rows = 2
            columns = 4
            for i in range(n_step):
                fig = plt.figure(figsize=(7, 7))
                print(states.shape)
                for z in range(h_step):
                    fig.add_subplot(rows, columns, z+1)
                    print("LSTM " + str(i + 1) + str(playerstatus[i]))
                    plt.imshow(states[i][z])
                    plt.axis('off')
                plt.show()
        return a_loss, c_loss


class replay():
    '''This class used as a buffer for the replay memory. Batches of 4 memory units are extracted from the memory and feeded at the network.
    Each memory unit contains 4 observations 3 for the history input at the lstms and one last observation. For the last observation also
     included the value it posses and the action which follow up based on that observation.'''
    def __init__(self,memory_size):
        self.memory_size = memory_size
        self.replay_buffer = []

    def take_batch(self,batch_size):
        '''This function returns the batch for the training of the network'''
        batch_start_boolean = True
        if len(self.replay_buffer) >= batch_size:
            while batch_start_boolean:
                batch_start = random.randint(0,len(self.replay_buffer)-1)
                if len(self.replay_buffer) - batch_start >= batch_size:
                    batch_start_boolean = False   
        return self.replay_buffer[batch_start:(batch_start+batch_size)]

    def write_memory(self,memory):
        '''This function writes one memory unit at the replay memory'''
        if  len(self.replay_buffer) >= self.memory_size:
            del self.replay_buffer[0]
        self.replay_buffer.append(memory)
    
    def create_memory(self,state,playerStatus, gameText, reward,action):
        '''This function creates one memory unit. Each memory unit contains 4 observations 3 for the history input at the lstms 
        and one last observation. For the last observation also included the value it posses and the action which follow up based on that observation..'''
        memory = []
        memory.append(state)
        memory.append(playerStatus)
        memory.append(gameText)
        memory.append(reward)
        memory.append(action)
        self.write_memory(memory)

    

# tf.random.set_seed(39999)
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\steps.txt'):
    f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\steps.txt','r')
    total_steps = int(f.read())
    f.close()
agentoo7 = agent()
replay_memory = replay(memory_size)
f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\episodes.txt','r')
episodes_text = int(f.read())
f.close()
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\ actor_model2.data-00000-of-00001'):
    print("actor model is loaded")
    agentoo7.actor.built = True
    agentoo7.actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\ actor_model2')
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\ critic_model2.data-00000-of-00001'):
    print("critic model is loaded")
    agentoo7.critic.built = True
    agentoo7.critic.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\ critic_model2')
episode = 10000
ep_reward = []
total_avgr = []
dfrewards = []
game = GamePAI(1,'Connan',444,444,screenfactor,True,episodes_text,False,seeded,0)
game_No = episodes_text
for s in range(episodes_text,episode):
    game_No = game_No + 1
    done = False
    state,playerStatus, gameText = game.initialGameState()
    rewards = []
    total_reward = 0
    all_aloss = []
    all_closs = []
    replay_memory.replay_buffer = []
    steps = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print('The record starts')
                    record = True
                if event.key == pygame.K_u:
                    print('The record stops')
                    record = False
        steps += 1
        state,playerStatus,gameText = agentoo7.preprocess0(state,playerStatus,gameText)
        buffer_state,buffer_playerStatus,buffer_gameText = state,playerStatus,gameText
        action = agentoo7.act(state,playerStatus,gameText)
        next_state, next_playerStatus, next_gameText,reward,done  = game.playerAction(action)
        rewards.append(reward)
        # value_of_state = agentoo7.preprocessVS(rewards,0.9)
        # print(rewards,value_of_state)
        replay_memory.create_memory(buffer_state,buffer_playerStatus,buffer_gameText,reward,action)
        action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
        print(action_name[action],reward,game.cave,steps,s,buffer_state.shape)
        
        total_reward += reward
        state = next_state
        playerStatus = next_playerStatus
        gameText = next_gameText
        

        if done:
            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,0)

        if steps%1000 == 0:
            print(steps,total_reward,action_name[action],game.cave)

        if s <= 2000: 
            if steps >= 500 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,0)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,0)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True
        if s > 2000: 
            if steps >= 5000 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,0)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,0)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True

        if steps > n_step + h_step:
            gamma = 0.99
            x_1,x_2,x_3,x_4 = replay_memory.replay_buffer[-4], replay_memory.replay_buffer[-3], replay_memory.replay_buffer[-2], replay_memory.replay_buffer[-1]
            if replay_memory.replay_buffer[-4][0].shape[1] > n_step:
                if not done:
                    G = 0
                    train_rewards = [x_1[3],x_2[3],x_3[3],x_4[3]]
                    for i in range(n_step):
                        G = G + (gamma**(i))*train_rewards[i]
                    train_state_t_n,train_playerstatus_T_t_n,train_gametext_t_n = x_4[0],x_4[1],x_4[2]
                    train_value_of_state = agentoo7.critic(train_state_t_n,train_playerstatus_T_t_n,train_gametext_t_n,training=True)
                    G = G + (gamma**n_step)*train_value_of_state
                    train_state,train_playerstatus_T,train_gametext = x_1[0],x_1[1],x_1[2]
                    al,cl = agentoo7.learn(train_state,train_playerstatus_T,train_gametext, G, x_1[4])
                    print('Actor losses ' + str(al) + ' Critic Losses ' + str(cl) +' Discounted reward ' + str(G) + ' for action ' + action_name[x_1[4]])
            if done:
                train_state_all = [x_1[0],x_2[0],x_3[0],x_4[0]]
                train_playerstatus_all = [x_1[1],x_2[1],x_3[1],x_4[1]]
                train_gametex_all = [x_1[2],x_2[2],x_3[2],x_4[2]]
                train_rewards = [x_1[3],x_2[3],x_3[3],x_4[3]]
                train_actions = [x_1[4],x_2[4],x_3[4],x_4[4]]
                for i in range(n_step):
                    G = 0
                    p = 0
                    for z in range(i,n_step):
                        G = G + (gamma**(p))*train_rewards[z]
                        p += 1
                    train_state,train_playerstatus_T,train_gametext = train_state_all[i], train_playerstatus_all[i], train_gametex_all[i]
                    if train_state.shape[1] > 4:
                        al,cl = agentoo7.learn(train_state,train_playerstatus_T,train_gametext, G, train_actions[i])
                        # print('I learn ' + action_name[train_actions[i]])
                        print('Actor losses ' + str(al) + ' Critic Losses ' + str(cl) +' Discounted reward ' + str(G) + ' for action ' + action_name[train_actions[i]])
        if done: 
            total_steps = total_steps + steps
            if s%100 == 0 and s != episodes_text:
                agentoo7.actor.save_weights('.\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\ actor_model2')
                agentoo7.critic.save_weights('.\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\ critic_model2')
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\episodes.txt','w')
                f.write(str(episodes_text + 100))
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\episodes.txt','r')
                episodes_text = int(f.read())
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_DR_2_model\steps.txt','w')
                f.write(str(total_steps))
                f.close()
                # f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\learning_rate.txt','w')
                # f.write(str(np.array(agentoo7.a_opt._decayed_lr(tf.float32), dtype=np.float32)))
                # f.close()
            # print(np.array(agentoo7.a_opt._decayed_lr(tf.float32), dtype=np.float32))
            # print(total_steps,agentoo7.a_opt._decayed_lr(tf.float32))
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
            episodeStat = [s,total_reward,avg_reward]
            dfrewards.append(episodeStat)
dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
dfrewards.to_excel(r'D:\ekpa\diplomatiki\playerActor2CNNnStepwithtext.xlsx')
