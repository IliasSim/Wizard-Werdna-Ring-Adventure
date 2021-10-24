from GamePAI import GamePAI
import pygame
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import GameSettings as settings
import time
import pandas as pd
import gc
import sys


class critic(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        self.l2 = tf.keras.layers.Conv2D(64,(4,4),2,activation = 'relu')
        self.l3 = tf.keras.layers.Flatten()
        self.l4 = tf.keras.layers.Dense(256,activation = 'relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self,input_data):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.l3(x)
        x= self.l4(x)
        v = self.v(x)
        return v

class actor(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv2D(32,(8,8),4,activation = 'relu',input_shape=(148,148,3))
        self.l2 = tf.keras.layers.Conv2D(64,(4,4),2,activation = 'relu')
        self.l3 = tf.keras.layers.Flatten()
        self.l4 = tf.keras.layers.Dense(512,activation = 'relu')
        self.a = tf.keras.layers.Dense(9, activation ='softmax')

    def call(self,input_data):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.l3(x)
        x= self.l4(x)
        a = self.a(x)
        return a

class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None
    
    def act(self,state):
        prob = self.actor(state)
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


    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss



    def learn(self, state, action, reward, next_state, done):
        #state = np.array([state])
        #next_state = np.array([next_state])
        #self.gamma = tf.convert_to_tensor(0.99, dtype=tf.double)
        #d = 1 - done
        #d = tf.convert_to_tensor(d, dtype=tf.double)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state, training=True)
            v =  self.critic(state,training=True)
            vn = self.critic(next_state, training=True)
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td**2
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

agentoo7 = agent()
steps = 1000
total_reward = 0
start_time = time. time()
dfrewards = []
game = GamePAI(1,'Connan',444,444,1,True,0,False)
for s in range(steps):
    done = False
    screen = game.screen
    array = pygame.surfarray.array3d(screen)
    array = array.swapaxes(0,1)
    array = array[:settings.mapHeigth,:settings.mapWidth]
    array =  np.expand_dims(array, axis=0)
    state = array/255
    #total_reward = 0
    all_aloss = []
    all_closs = []
    act = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        #env.render()
        act += 1
        action = agentoo7.act(state)
        #print(action)
        next_state, reward = game.playerAction(action)
        done = game.gameOver(s)
        aloss, closs = agentoo7.learn(state, action, reward, next_state, done)
        all_aloss.append(aloss)
        all_closs.append(closs)
        state = next_state
        #print(game.cave)
        if act% 1000 == 0:
            print(act,reward,len(settings.enemies),action,game.cave)

        if act >= 1000 and game.cave < 2:
            noVideo = True
            if s% 100 == 0:
                game.__init__(1,'Connan',444,444,1,True,s,False)
                noVideo = False
            if noVideo:
                game.__init__(1,'Connan',444,444,1,True,s,False)
            gc.collect()
            done = True
        
        #total_reward += reward
        if done == True:
            act = 0
            total_reward += reward 
            mean_reward = total_reward/(s+1)
            episodeStat = [s,reward,mean_reward]
            dfrewards.append(episodeStat)
            print('end of episode ' + str(s) + ' episode rewards ' + str(reward) + ' mean reward:' + str(mean_reward))
dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
dfrewards.to_excel(r'D:\ekpa\diplomatiki\agent2CNNrewardsNormalize_to_one.xlsx')
print("--- %s seconds ---" % (time. time() - start_time))