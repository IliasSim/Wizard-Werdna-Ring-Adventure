from GamePAI_tensorflow import GamePAI
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
h_step = 4
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = True
input = (148,148,3)
record = False
memory_size = 100000
batch_size = 12
agent_i = 0
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

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
        self.actor = actor()
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
       
        if len(self.buffer_State) >= self.buffer_size:
            del self.buffer_State[0]
        self.buffer_State.append(state)
        state = np.array(self.buffer_State,dtype=np.float32)
        state = np.expand_dims(state, axis=0)

        if len(self.buffer_playerStatus) >= self.buffer_size:
            del self.buffer_playerStatus[0]
        self.buffer_playerStatus.append(playerstatus)
        playerstatus = np.array(self.buffer_playerStatus, dtype=np.float32)
        playerstatus = np.expand_dims(playerstatus, axis=0)

        if len(self.buffer_text) >= self.buffer_size:
            del self.buffer_text[0]
        self.buffer_text.append(gameText)
        gameText = np.array(self.buffer_text, dtype=np.float32)
        gameText = np.expand_dims(gameText, axis=0)
        return state,playerstatus,gameText



# tf.random.set_seed(39999 + i)
agentoo7 = agent()
total_steps = 0
episodes_text = 0
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_16_h_step_truncate_steped_lstm\ actor_model2.data-00000-of-00001'):
    print("actor model is loaded")
    agentoo7.actor.built = True
    agentoo7.actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_trueonline_16_h_step_truncate_steped_lstm\ actor_model2')
episode = 300
ep_reward = []
total_avgr = []
dfrewards = []
game = GamePAI(1,'Connan',444,444,screenfactor,True,episodes_text,False,seeded,agent_i)
game_No = episodes_text
for s in range(episodes_text,episode):
    game_No = game_No + 1
    done = False
    state,playerStatus, gameText = game.initialGameState()
    rewards = []
    total_reward = 0
    # all_aloss = []
    # all_closs = []
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
        action = agentoo7.act(state,playerStatus,gameText)
        next_state, next_playerStatus, next_gameText,reward,done  = game.playerAction(action)
        rewards.append(reward)
        action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
        print(agent_i,action_name[action],game.cave,steps)
        total_reward += reward
        state = next_state
        playerStatus = next_playerStatus
        gameText = next_gameText
        
        cave = game.cave

        if steps%5000 == 0:
            done = True
            print(steps,total_reward,action_name[action],game.cave)

        if done:
            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,agent_i)

               
        if done:
            total_steps = total_steps + steps
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
            
