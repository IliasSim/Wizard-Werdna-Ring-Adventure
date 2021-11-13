from GamePAI_image_lstm_only import GamePAI
import pygame
import numpy as np
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import gc
import tensorflow.keras.losses as kls
from os.path import exists
import sys

featuresCNN1 = 32
CNN1Shape = 4
CNN1Step = 4
featuresCNN2 = 32
CNN2Shape = 2
CNN2Step = 2
featuresCNN3 = 64
CNN3Shape = 3
CNN3Step = 1
denseLayerN = 256
denseLayerNL_2 = 32
denseLayerNL_3= 64
denseLayerNL_21 = 64
denseLayerNL_31 = 128
dropoutRate1 = 0.3
dropoutRate2 = 0.3 
h_step = 8
n_step = 4
screenfactor = 1
input = (148,148,3)
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
class critic(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        # self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(64,(4,4),(1,1),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step))
        # self.l2 = tf.keras.layers.Conv2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step))
        # self.l3 = tf.keras.layers.Flatten()
        # self.l4 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        self.l1 = tf.keras.layers.Conv3D(featuresCNN1,(1,CNN1Shape,CNN1Shape),(1,CNN1Step,CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv3D(featuresCNN2,(1,CNN2Shape,CNN2Shape),(1,CNN2Step,CNN2Step),activation = 'relu')
        # self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((h_step,10368))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        # self.l5 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        # self.l6 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        # self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        # self.l7 = tf.keras.layers.Dense(denseLayerNL_21,activation = 'relu')
        # self.l8 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        # self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        # self.l9 = tf.keras.layers.Dense(denseLayerNL_31,activation = 'relu')
        # self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        # self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        #self.drop1 = tf.keras.layers.Dropout(dropoutRate1)
        #self.drop2 = tf.keras.layers.Dropout(dropoutRate2)
        #self.conc = tf.keras.layers.concatenate([self.l4,self.l1_2],axis = -1)
        #self.conc3 = tf.keras.layers.concatenate([self.l1_3,self.conc],axis = -1)
        self.v = tf.keras.layers.Dense(1, activation =None)
      

    def call(self,input_data):
        x = self.l1(input_data)
        x = self.l2(x)
        # x = self.l3(x)
        x = self.l4(x)
        x= self.l5(x)
        # y = self.l6(input_data1)
        # y = self.l7(y)
        # z = self.l8(input_data2)
        # z = self.l9(z)
        # h = self.conc1((y,z))
        # g = self.conc2((h,x))
        v = self.v(x)
        return v

class actor(tf.keras.Model):
    '''This class creates the model of the actor part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
       # self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(64,(4,4),(1,1),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step))
        # self.l2 = tf.keras.layers.Conv2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step))
        # self.l3 = tf.keras.layers.Flatten()
        # self.l4 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        self.l1 = tf.keras.layers.Conv3D(featuresCNN1,(1,CNN1Shape,CNN1Shape),(1,CNN1Step,CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv3D(featuresCNN2,(1,CNN2Shape,CNN2Shape),(1,CNN2Step,CNN2Step),activation = 'relu')
        # self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((h_step,10368))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        # self.l5 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        # self.l6 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        # self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        # self.l7 = tf.keras.layers.Dense(denseLayerNL_21,activation = 'relu')
        # self.l8 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        # self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        # self.l9 = tf.keras.layers.Dense(denseLayerNL_31,activation = 'relu')
        # self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        # self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        #self.drop1 = tf.keras.layers.Dropout(dropoutRate1)
        #self.drop2 = tf.keras.layers.Dropout(dropoutRate2)
        #self.conc = tf.keras.layers.concatenate([self.l4,self.l1_2],axis = -1)
        #self.conc3 = tf.keras.layers.concatenate([self.l1_3,self.conc],axis = -1)
        self.a = tf.keras.layers.Dense(9, activation ='softmax')
      

    def call(self,input_data,):
        x = self.l1(input_data)
        x = self.l2(x)
        # x= self.l3(x)
        x = self.l4(x)
        x= self.l5(x)
        # y = self.l6(input_data1)
        # y = self.l7(y)
        # y = self.drop1(y)
        # z = self.l8(input_data2)
        # z = self.l9(z)
        # z = self.drop2(z)
        # h = self.conc1((y,z))
        # e = self.add((y,d))
        # g = self.conc2((h,x))
        a = self.a(x)
        return a

class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9)
        self.a_opt = tf.keras.optimizers.Adam(lr_schedule)
        self.c_opt = tf.keras.optimizers.Adam(lr_schedule)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None
    
    def act(self,state,):
        prob = self.actor(state,)
        print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        #print(action)
        return int(action.numpy()[0])
        # action = np.random.choice([i for i in range(env.action_space.n)], 1, p=prob[0])
        # log_prob = tf.math.log(prob[0][action]).numpy()
        # self.log_prob = log_prob[0]
        # #print(self.log_prob)
        # return action[0]
    

    def preprocess1(self,states, actions, rewards, gamma):
        discnt_rewards = []
        sum_reward = 0        
        rewards.reverse()
        for r in rewards:
            sum_reward = r + gamma*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
        states = np.array(states, dtype=np.float32)
        states =  np.squeeze(states,axis = 1)
        # atus = np.array(playerstatus,dtype=np.int32)
        # playerstatus = np.squeeze(playerstatus,axis=1)
        # gameTexts = np.array(gameTexts,dtype=np.int32)
        # gameTexts = np.expand_dims(gameTexts, axis=0)
        # print(gameTexts.shape,gameTexts,playerstatus,playerstatus.shape)
        # gameTexts = np.squeeze(gameTexts,axis = 0)
        # gameTexts = np.squeeze(gameTexts,axis = 1)
        # print(gameTexts.shape,playerstatus.shape)
        #gameTexts = np.expand_dims(gameTexts, axis=0)
        #print(gameTexts.shape,playerstatus.shape,states.shape)
        actions = np.array(actions, dtype=np.int32)
        discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
        return states, actions, discnt_rewards

    def actor_loss(self, probs, actions, td): 
        probability = []
        log_probability= []
        for pb,a in zip(probs,actions):
          dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
          log_prob = dist.log_prob(a)
          prob = dist.prob(a)
          probability.append(prob)
          log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)

        p_loss= []
        e_loss = []
        td = td.numpy()
        #print(td)
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
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        #print(loss)
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            # print(discnt_rewards)
            # print(v)
            #print(td.numpy())
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

#tf.random.set_seed(336699)
agentoo7 = agent()
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\episodes.txt'):
    f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\episodes.txt','r')
    episodes_text = int(f.read())
    f.close()
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\ actor_model.data-00000-of-00001'):
    print("actor model is loaded")
    agentoo7.actor.built = True
    agentoo7.actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\ actor_model')
if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\ critic_model.data-00000-of-00001'):
    print("critic model is loaded")
    agentoo7.critic.built = True
    agentoo7.critic.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\ critic_model')
episode = 10000
ep_reward = []
total_avgr = []
dfrewards = []
game = GamePAI(1,'Connan',444,444,screenfactor,True,0,False)
for s in range(episode):
    game_No = s + 1
    done = False
    state = game.initialGameState()
    #gameText = agentoo7.preprocess(gameText)
    total_reward = 0
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    # playerstatus = []
    gameTexts = []
    actions = []
    steps = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        steps += 1
        action = agentoo7.act(state)
        #done = game.gameOver(s)
        next_state,reward,done  = game.playerAction(action)
        action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
        print(action_name[action],reward,game.cave)
        if done:
            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False)
        #next_gameText = agentoo7.preprocess(next_gameText)
        #done = game.gameOver(s)
        rewards.append(reward)
        states.append(state)
        # playerstatus.append(playerStatus)
        # gameTexts.append(gameText)
        #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
        actions.append(action)
        state = next_state
        # playerStatus = next_playerStatus
        # gameText = next_gameText
        total_reward += reward
        if steps%1000 == 0:
            print(steps,total_reward,action_name[action],game.cave)

        if s + episodes_text <= 2000: 
            if steps >= 2000 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True
        if s + episodes_text > 2000: 
            if steps >= 5000 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True

        if steps%h_step == 0 or done:
            states, actions, discnt_rewards = agentoo7.preprocess1(states, actions, rewards, 0.99)
            # print(discnt_rewards)
            if states.shape[0] > n_step:
                states = states[-n_step:]
                # playerstatus = playerstatus[-n_step:]
                # gameTexts = gameTexts[-n_step:]
                actions = actions[-n_step:]
                discnt_rewards = discnt_rewards[-n_step:]
            if states.shape[0] <= n_step:
                discnt_rewards = discnt_rewards[-states.shape[0]:]
            # print(discnt_rewards)
            al,cl = agentoo7.learn(states, actions, discnt_rewards) 
            states = []
            # playerstatus = []
            gameTexts = []
            actions = []
            rewards = []
            # print(f"al{al}") 
            # print(f"cl{cl}")
    
        if done:
            if s%100 == 0 and s != 0:
                agentoo7.actor.save_weights('.\model_LSTM_image_11_11_21\ actor_model')
                agentoo7.critic.save_weights('.\model_LSTM_image_11_11_21\ critic_model')
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\episodes.txt','w')
                f.write(str(episodes_text + 100))
                f.close()
                f = open('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_LSTM_image_11_11_21\episodes.txt','r')
                episodes_text = int(f.read())
                f.close()
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
            episodeStat = [s,total_reward,avg_reward]
            dfrewards.append(episodeStat)
dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
dfrewards.to_excel(r'D:\ekpa\diplomatiki\playerActor2CNNnStepwithtext.xlsx')
