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
denseLayerNL_21 = 64
denseLayerNL_31 = 128 
h_step = 8
n_step = 4
screenfactor = 1
decay_steps = 10000
seeded = True
input = (148,148,3)
record = False
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
        # self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((-1,161312))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        # self.l6 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        # self.l8 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        self.v = tf.keras.layers.Dense(1, activation =None)
      

    def call(self,input_data,input_data1,input_data2):
        x = self.l1(input_data)
        x = self.l2(x)
        # x = self.l3(x)
        x = self.l4(x)
        x= self.l5(x)
        # y = self.l6(input_data1)
        y = self.l7(input_data1)
        # z = self.l8(input_data2)
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
        # self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((-1,161312))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        # self.l6 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        # self.l8 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        self.a = tf.keras.layers.Dense(9, activation ='softmax')
      

    def call(self,input_data,input_data1,input_data2):
        x = self.l1(input_data)
        x = self.l2(x)
        # x= self.l3(x)
        x = self.l4(x)
        x= self.l5(x)
        # y = self.l6(input_data1)
        y = self.l7(input_data1)
        # z = self.l8(input_data2)
        z = self.l9(input_data2)
        h = self.conc1((y,z))
        g = self.conc2((h,x))
        a = self.a(g)
        return a

class agent():
    def __init__(self,initial_learning_rate, gamma = 0.99):
        self.gamma = gamma
        self.initial_learning_rate = initial_learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=0.9)
        self.a_opt = tf.keras.optimizers.Adam(lr_schedule)
        self.c_opt = tf.keras.optimizers.Adam(lr_schedule)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None
        self.buffer_State = []
        self.buffer_playerStatus = []
        self.buffer_text = []
        self.buffer_reward = []
        self.buffer_size = 7

    def act(self,state,playerstatus,gameText):
        prob = self.actor(state,playerstatus,gameText)
        print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def preprocess0(self, state,playerstatus,gameText):
        if len(self.buffer_State) > self.buffer_size:
            del self.buffer_State[0]
        self.buffer_State.append(state)
        state = np.array(self.buffer_State)
        state = np.expand_dims(state, axis=0)

        if len(self.buffer_playerStatus) > self.buffer_size:
            del self.buffer_playerStatus[0]
        self.buffer_playerStatus.append(playerstatus)
        playerstatus = np.array(self.buffer_playerStatus, dtype=np.float32)
        playerstatus = np.expand_dims(playerstatus, axis=0)

        if len(self.buffer_text) > self.buffer_size:
            del self.buffer_text[0]
        self.buffer_text.append(gameText)
        gameText = np.array(self.buffer_text, dtype=np.float32)
        gameText = np.expand_dims(gameText, axis=0)    
        # print(state.shape,playerstatus.shape)
        return state,playerstatus,gameText

    def preprocess1(self, state,playerstatus,gameText, rewards, gamma):
        discnt_rewards = []
        sum_reward = 0        
        rewards.reverse()
        for r in rewards:
            sum_reward = r + gamma*sum_reward
            discnt_rewards.append(sum_reward)
        rewards.reverse()
        discnt_rewards.reverse()
        discnt_rewards = discnt_rewards[-n_step:]
        state = np.array(state)
        state =  np.squeeze(state,axis = 1)
        playerstatus = np.array(playerstatus)
        playerstatus =  np.squeeze(playerstatus,axis = 1)
        gameText = np.array(gameText)
        gameText =  np.squeeze(gameText,axis = 1)
        return  state,playerstatus,gameText,discnt_rewards

    def actor_loss(self, probs, actions, td): 
        probability = []
        log_probability= []
        # for pb,a in zip(probs,actions):
        dist = tfp.distributions.Categorical(probs, dtype=tf.float32)
        log_prob = dist.log_prob(actions)
        prob = dist.prob(actions)
        probability.append(prob)
        log_probability.append(log_prob)

        p_loss= []
        e_loss = []
        td = td.numpy()
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
        loss = -p_loss - 0.0001 * e_loss
        return loss

    def learn(self, states,playerstatus,gameTexts, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        # print(states.shape,playerstatus.shape,gameTexts.shape)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states,playerstatus,gameTexts, training=True)
            # print(states.shape,playerstatus.shape,gameTexts.shape,p)
            v =  self.critic(states,playerstatus,gameTexts,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        
        if record:
            rows = 2
            columns = 4
            
            # print(states[0].shape[0])
            for i in range(n_step):
                fig = plt.figure(figsize=(7, 7))
                print(states.shape)
                for z in range(h_step):
                    # print(z)
                    fig.add_subplot(rows, columns, z+1)
                    print("LSTM " + str(i + 1) + str(playerstatus[i]))
                    #print(states[i][z].shape)
                    plt.imshow(states[i][z])
                    plt.axis('off')
                    # plt.title(action_name[actions[i]])
                plt.show()
        return a_loss, c_loss

if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\steps.txt'):
    f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\steps.txt','r')
    total_steps = int(f.read())
    f.close()
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\learning_rate.txt'):
    f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\learning_rate.txt','r')
    initial_learning_rate = float(f.read())
    f.close()
# initial_learning_rate = 1e-5*0.9**(total_steps/(h_step*decay_steps))
if initial_learning_rate == 0:
    initial_learning_rate = 1e-5
agentoo7 = agent(initial_learning_rate)
f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\episodes.txt','r')
episodes_text = int(f.read())
f.close()
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\ actor_model.data-00000-of-00001'):
    print("actor model is loaded")
    agentoo7.actor.built = True
    agentoo7.actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\ actor_model')
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\ critic_model.data-00000-of-00001'):
    print("critic model is loaded")
    agentoo7.critic.built = True
    agentoo7.critic.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\ critic_model')
episode = 10000 - episodes_text
ep_reward = []
total_avgr = []
dfrewards = []
game = GamePAI(1,'Connan',444,444,screenfactor,True,episodes_text,False,seeded)
game_No = episodes_text
for s in range(episode):
    game_No = game_No + 1
    done = False
    state,playerStatus, gameText = game.initialGameState()
    # print(state.shape)
    state,playerStatus, gameText = agentoo7.preprocess0(state,playerStatus, gameText)
    total_reward = 0
    rewards = []
    train_actions = []
    train_states = []
    train_playerstatus = []
    train_gametexts = []
    all_aloss = []
    all_closs = []
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
        action = agentoo7.act(state,playerStatus,gameText)
        next_state,reward, next_playerStatus, next_gameText,done  = game.playerAction(action)
        action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
        # print(reward)
        print(action_name[action],reward,game.cave)
        next_state,next_playerStatus,next_gameText = agentoo7.preprocess0(next_state,next_playerStatus,next_gameText)
        state = next_state
        playerStatus = next_playerStatus
        gameText = next_gameText
        rewards.append(reward)
        total_reward += reward
        if len(train_playerstatus) >= n_step:
            train_actions = []
            train_states = []
            train_playerstatus = []
            train_gametexts = []
        train_actions.append(action)
        train_states.append(state)
        train_playerstatus.append(playerStatus)
        train_gametexts.append(gameText)
        if done:
            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)

        if steps%1000 == 0:
            print(steps,total_reward,action_name[action],game.cave)

        if s + episodes_text <= 2000: 
            if steps >= 2000 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True
        if s + episodes_text > 2000: 
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
        if steps > h_step:
            if steps%n_step == 0:
                train_states,train_playerstatus,train_gametexts,discnt_rewards= agentoo7.preprocess1(train_states,train_playerstatus,train_gametexts, rewards, 0.99)
                al,cl = agentoo7.learn(train_states,train_playerstatus,train_gametexts, train_actions, discnt_rewards)
                # print(train_state[0].shape)
                # actions, discnt_rewards = agentoo7.preprocess2(actions, rewards, 0.99)
                # for i in range(n_step,h_step):
                    # print(states[i].shape,playerstatus[i].shape,gameTexts[i].shape)
                    # al,cl = agentoo7.learn(states[i],playerstatus[i],gameTexts[i], actions[i], discnt_rewards[i])
                    # print("epoch " + str(i) +" " +  str(steps) + str(agentoo7.a_opt._decayed_lr(tf.float32)))
                # states = states[n_step:]
                # print(states[0].shape)
                # playerstatus = playerstatus[n_step:]
                # gameTexts = gameTexts[n_step:]
                # actions = actions[n_step:]
                # actions = actions.tolist()
        
        if done:
            total_steps = total_steps + steps
            agentoo7.last_discnt_rewards = 0
            if s%100 == 0 and s != 0:
                agentoo7.actor.save_weights('.\playerActorLSTM_Imp_15_12_21\ actor_model')
                agentoo7.critic.save_weights('.\playerActorLSTM_Imp_15_12_21\ critic_model')
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\episodes.txt','w')
                f.write(str(episodes_text + 100))
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\episodes.txt','r')
                episodes_text = int(f.read())
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\steps.txt','w')
                f.write(str(total_steps))
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\playerActorLSTM_Imp_15_12_21\learning_rate.txt','w')
                f.write(str(np.array(agentoo7.a_opt._decayed_lr(tf.float32), dtype=np.float32)))
                f.close()
            print(np.array(agentoo7.a_opt._decayed_lr(tf.float32), dtype=np.float32))
            print(total_steps,agentoo7.a_opt._decayed_lr(tf.float32))
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
            episodeStat = [s,total_reward,avg_reward]
            dfrewards.append(episodeStat)
dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
dfrewards.to_excel(r'D:\ekpa\diplomatiki\playerActor2CNNnStepwithtext.xlsx')
