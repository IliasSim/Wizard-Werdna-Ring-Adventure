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
h_step = 4
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = True
input = (148,148,3)
record = False
memory_size = 250000
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
        x= self.l5(x)
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
    Also contains functions for the training of the networks and the preprocess of the input data'''
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
        self.actor_e = actor()
        self.critic_e = critic()
        self.actor_c = actor()
        self.critic_c = critic()
        print(self.actor,self.actor_e)
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
        '''This function cretaes the history of observation for the input to the NN during the agent interaction with the environment.'''
       
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

    def preprocess1(self,batch,gamma):
        '''This function take at the end of each episode the total memory containing the memory unit(observation, action and reward) and use them to create
        a package of three observation for history, one last observation and the action that followed the last observation. It calculate for each last observation 
        and the discounted reward using the rewards from the last observation until the end of the episode. After the function output is used for 
        the training of the network.'''
        state = []
        playerstatus = []
        gameText = []
        discounted_rewards = []
        rewards = []
        actions = []
        for i in range(h_step-1,len(batch)):
            actions.append(batch[i][4])

        for i in range(h_step-1,len(batch)):
            rewards.append(batch[i][3])

        for i in range(h_step-1,len(batch)):
            sum_rewards = 0
            power = 0
            for z in range(i,len(batch)):
                reward = batch[z][3]*(gamma**power)
                sum_rewards += reward
                power += 1
            discounted_rewards.append(sum_rewards)
        for z in range(3,len(batch)):
            print(batch[z][3])
        print(discounted_rewards,len(discounted_rewards))

        for i in range(h_step-1,len(batch)):
            state_batch = []
            playerstatus_batch = []
            gameText_batch = []
            for z in range(i - 3,i + 1):
                state_batch.append(batch[z][0])
                playerstatus_batch.append(batch[z][1])
                gameText_batch.append(batch[z][2])

            state_batch = np.array(state_batch,dtype=np.float32)
            playerstatus_batch = np.array(playerstatus_batch,dtype=np.float32)
            gameText_batch = np.array(gameText_batch,dtype=np.float32)
            
            state.append(state_batch)
            playerstatus.append(playerstatus_batch)
            gameText.append(gameText_batch)
            
        state = np.array(state,dtype=np.float32)
        playerstatus = np.array(playerstatus,dtype=np.float32)
        gameText = np.array(gameText,dtype=np.float32)        
        return  state,playerstatus,gameText,discounted_rewards,actions

    def preprocess2(self,train_state,train_playerstatus,train_gameText):
        '''This function create the rigth dimension for the data that are gone to be used by the learn function'''
        train_state = np.expand_dims(train_state,axis=0)
        train_playerstatus = np.expand_dims(train_playerstatus,axis = 0)
        train_gameText = np.expand_dims(train_gameText,axis = 0)
        return train_state,train_playerstatus,train_gameText

    def actor_loss(self, probs, actions, td,oneway):
        '''This function calculate actor NN losses. Which is negative of Log probability of action taken multiplied 
        by temporal difference used in q learning.'''
        print('probs ' + str(probs))
        probability = []
        log_probability= []
        if oneway:
            dist = tfp.distributions.Categorical(probs, dtype=tf.float32)
            # print(dist)
            # log_prob = tf.math.log(probs[0][actions])
            dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
            log_prob = dist.log_prob(actions)
            print('actions ' + str(actions))
            print('log_prob ' + str(log_prob))
            prob = dist.prob(actions)
            probability.append(prob)
            log_probability.append(log_prob)
        if not oneway:
            for pb,a in zip(probs,actions):
                dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
                log_prob = dist.log_prob(a)
                print('actions' + str(a))
                print('log_prob ' + str(log_prob))
                prob = dist.prob(a)
                probability.append(prob)
                log_probability.append(log_prob)

        print('probability ' + str(probability))
        print('log_probability ' + str(log_probability))

        p_loss= []
        e_loss = []
        td = td.numpy()
        print('td ' + str(td))
        for pb, t, lpb in zip(probability, td, log_probability):
                        t =  tf.constant(t)
                        policy_loss = tf.math.multiply(lpb,t)
                        entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))
                        p_loss.append(policy_loss)
                        e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        print('p_loss ' + str(p_loss))
        print('e_loss ' + str(e_loss))
        loss = -p_loss - 0.0001 * e_loss
        print('loss' + str(loss))
        return loss

    def learnC(self, states,playerstatus,gameTexts, G,actions,oneway):
        '''This function is used for the training of the network, it also contain a code chunk for the depiction of the image used for the training. 
        For critic loss, we took a naive way by just taking the square of the temporal difference.'''
        G = tf.reshape(G, (len(G),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states,playerstatus,gameTexts, training=True)
            v =  self.critic(states,playerstatus,gameTexts,training=True)
            v = tf.reshape(v, (len(v),))
            # print(p)
            print(v)
            td = tf.math.subtract(G, v)
            a_loss = self.actor_loss(p, actions, td,oneway)
            c_loss = 0.5*kls.mean_squared_error(G, v)
            # print(a_loss,c_loss)
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

    def learn(self, states,playerstatus,gameTexts, G,actions,oneway):
        '''This function is used for the training of the network, it also contain a code chunk for the depiction of the image used for the training. 
        For critic loss, we took a naive way by just taking the square of the temporal difference.'''
        # discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor_c(states,playerstatus,gameTexts)
            v = self.critic_c(states,playerstatus,gameTexts)
            v = tf.reshape(v, (len(v),))
            # print(p)
            print(v)
            td = tf.math.subtract(G, v)
            a_loss = self.actor_loss(p, actions, td,oneway)
            c_loss = 0.5*kls.mean_squared_error(G, v)
        grads1 = tape1.gradient(a_loss, self.actor_c.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic_c.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor_c.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic_c.trainable_variables))
        
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
        '''This function creates one memory unit. Each memory unit contains the last observation of the state and also 
        the action and the reward that follow.'''
        memory = []
        memory.append(state)
        memory.append(playerStatus)
        memory.append(gameText)
        memory.append(reward)
        memory.append(action)
        self.write_memory(memory)

    

tf.random.set_seed(39999)

if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\steps.txt'):
    f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\steps.txt','r')
    total_steps = int(f.read())
    f.close()
f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\episodes.txt','r')
episodes_text = int(f.read())
agentoo7 = agent()
game = GamePAI(1,'Connan',444,444,screenfactor,True,episodes_text,False,seeded)
state,playerStatus, gameText = game.initialGameState()
state,playerStatus,gameText = agentoo7.preprocess0(state,playerStatus,gameText)
agentoo7.actor(state,playerStatus,gameText)
agentoo7.actor_c(state,playerStatus,gameText)
agentoo7.actor_e(state,playerStatus,gameText)
agentoo7.critic(state,playerStatus,gameText)
agentoo7.critic_c(state,playerStatus,gameText)
agentoo7.critic_e(state,playerStatus,gameText)
del state,playerStatus,gameText
replay_memory = replay(memory_size)
f.close()
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\ actor_model2.data-00000-of-00001'):
    print("actor model is loaded")
    agentoo7.actor.built = True
    agentoo7.actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\ actor_model2')
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\ critic_model2.data-00000-of-00001'):
    print("critic model is loaded")
    agentoo7.critic.built = True
    agentoo7.critic.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\ critic_model2')
episode = 10000
ep_reward = []
total_avgr = []
dfrewards = []

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
        buffer_state,buffer_playerStatus,buffer_gameText = state,playerStatus,gameText
        state,playerStatus,gameText = agentoo7.preprocess0(state,playerStatus,gameText)
        # print(state.shape)
        action = agentoo7.act(state,playerStatus,gameText)
        next_state, next_playerStatus, next_gameText,reward,done  = game.playerAction(action)
        rewards.append(reward)
        # value_of_state = agentoo7.preprocessVS(rewards,0.9)
        # print(rewards,value_of_state)
        replay_memory.create_memory(buffer_state,buffer_playerStatus,buffer_gameText,reward,action)
        action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
        print(action_name[action],reward,game.cave,steps,s,len(replay_memory.replay_buffer))
        
        total_reward += reward
        state = next_state
        playerStatus = next_playerStatus
        gameText = next_gameText
        

        if done:
            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)

        if steps%1000 == 0:
            print(steps,total_reward,action_name[action],game.cave)

        if s <= 2000: 
            if steps >= 500 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True
        if s > 2000: 
            if steps >= 5000 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True

        if done:
            # h = hpy()
            # print(h.heap())
            
            # print(len(replay_memory.replay_buffer),total_steps,steps,(replay_memory.replay_buffer[0][0].nbytes + replay_memory.replay_buffer[0][1].nbytes
            # + replay_memory.replay_buffer[0][2].nbytes)*len(replay_memory.replay_buffer)/(1024*1024*1024))
            gamma = 0.9
            # batch = replay_memory.take_batch(batch_size = 100)
            train_states,train_playerstatus,train_gametexts,train_discounted_rewards,train_actions = agentoo7.preprocess1(replay_memory.replay_buffer,gamma)
            # print(train_states[].shape)
            # print(agentoo7.actor.get_weights())
            oneway = True
            if oneway:
                for i in range(len(train_states)):
                    agentoo7.actor_c.set_weights(agentoo7.actor.get_weights())
                    agentoo7.critic_c.set_weights(agentoo7.critic.get_weights())
                    train_states_T,train_playerstatus_T,train_gametexts_T = agentoo7.preprocess2(train_states[i],train_playerstatus[i],train_gametexts[i])
                    al,cl = agentoo7.learn(train_states_T,train_playerstatus_T,train_gametexts_T, train_discounted_rewards[i], train_actions[i],oneway)
                    agentoo7.actor_e.set_weights(agentoo7.actor_c.get_weights())
                    agentoo7.critic_e.set_weights(agentoo7.critic_c.get_weights())
                    print('Actor losses ' + str(al) + ' Critic Losses ' + str(cl) + str(train_discounted_rewards[i]) + ' for action ' + action_name[train_actions[i]])
                agentoo7.actor.set_weights(agentoo7.actor_e.get_weights())
                agentoo7.critic.set_weights(agentoo7.critic_e.get_weights())
            # print(train_states.shape,train_playerstatus.shape,train_gametexts.shape,len(train_discnt_rewards),len(train_actions))
            # train_states_T,train_playerstatus_T,train_gametexts_T = agentoo7.preprocess2(train_states,train_playerstatus,train_gametexts)
            if not oneway:
                al,cl = agentoo7.learnC(train_states,train_playerstatus,train_gametexts, train_discounted_rewards, train_actions,oneway)
                print(al,cl)
            
        if done: 
            total_steps = total_steps + steps
            if s%100 == 0 and s != episodes_text:
                agentoo7.actor.save_weights('.\modelLSTM_Imp_16_1_21_Monte_carlo\ actor_model2')
                agentoo7.critic.save_weights('.\modelLSTM_Imp_16_1_21_Monte_carlo\ critic_model2')
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\episodes.txt','w')
                f.write(str(episodes_text + 100))
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\episodes.txt','r')
                episodes_text = int(f.read())
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_16_1_21_Monte_carlo\steps.txt','w')
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
