from GamePAI_image_dense_only import GamePAI
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
        self.l1 = tf.keras.layers.Conv2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step))
        self.l2 = tf.keras.layers.Conv2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step))
        self.l3 = tf.keras.layers.Flatten()
        self.l5 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        self.v = tf.keras.layers.Dense(1, activation =None)
      

    def call(self,input_data):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.l3(x)
        x= self.l5(x)
        v = self.v(x)
        return v

class actor(tf.keras.Model):
    '''This class creates the model of the actor part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step))
        self.l2 = tf.keras.layers.Conv2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step))
        self.l3 = tf.keras.layers.Flatten()
        self.l5 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        self.a = tf.keras.layers.Dense(9, activation ='softmax')
      

    def call(self,input_data,):
        x = self.l1(input_data)
        x = self.l2(x)
        x= self.l3(x)
        x= self.l5(x)
        a = self.a(x)
        return a

class agent():
    def __init__(self,initial_learning_rate, gamma = 0.99):
        self.gamma = gamma
        self.initial_learning_rate = initial_learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.initial_learning_rate,
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
        return int(action.numpy()[0])
    

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

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
for i in range(4):   
    tf.random.set_seed(39999 + i)
    total_steps = 0
    agentoo7 = agent(1e-5)
    episodes_text = 0
    if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_Dense_image_11_11_21\ actor_model.data-00000-of-00001'):
        print("actor model is loaded")
        agentoo7.actor.built = True
        agentoo7.actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_Dense_image_11_11_21\ actor_model')
    if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_Dense_image_11_11_21\ critic_model.data-00000-of-00001'):
        print("critic model is loaded")
        agentoo7.critic.built = True
        agentoo7.critic.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\model_Dense_image_11_11_21\ critic_model')

    episode = 100
    ep_reward = []
    total_avgr = []
    dfrewards = []
    game = GamePAI(1,'Connan',444,444,screenfactor,True,episodes_text,False,True)
    game_No = episodes_text
    for s in range(episode):
        game_No = game_No + 1
        done = False
        state = game.initialGameState()
        total_reward = 0
        # all_aloss = []
        # all_closs = []
        # rewards = []
        # states = []
        # actions = []
        steps = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            steps += 1
            action = agentoo7.act(state)
        
            next_state,reward,done  = game.playerAction(action)
            action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
            print(action_name[action],reward,game.cave)
            cave = game.cave
            if done:
                game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,True)
            # rewards.append(reward)
            # states.append(state)
            # actions.append(action)
            state = next_state
            total_reward += reward

            if steps >= 2000 and game.cave < 2:
                noVideo = True
                if s% 100 == 0:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,True)
                    noVideo = False
                if noVideo:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,True)
                gc.collect()
                print(s,total_reward,game.cave)
                done = True
    
            if done:
                total_steps = total_steps + steps
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} episode is {} and avg reward is {} steps per episode{} total steps{} cave {}".format(s, total_reward, avg_reward,steps,total_steps,cave))
                episodeStat = [s,total_reward,avg_reward,steps,total_steps,cave]
                dfrewards.append(episodeStat)
    dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards','steps','total steps','cave'])
    file = 'D:\ekpa\diplomatiki\seeded_score\playerActorDense_image_11_11_21_test_' + str(i) + '.xlsx'
    dfrewards.to_excel(file)
