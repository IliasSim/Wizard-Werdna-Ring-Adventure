from GamePAI import GamePAI
import pygame
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import GameSettings as settings
import time
import pandas as pd
import gc
import tensorflow.keras.losses as kls
import sys

#seed = 42
#env.seed(seed)
#tf.random.set_seed(seed)
#np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

class critic(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        # self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        # self.l2 = tf.keras.layers.Conv2D(64,(4,4),2,activation = 'relu')
        self.l1 = tf.keras.layers.Conv2D(32,(3,3),(1,1),activation = 'relu',input_shape=(148,148,3))
        self.l2 = tf.keras.layers.Conv2D(16,(5,5),2,activation = 'relu')
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
        # self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        # self.l2 = tf.keras.layers.Conv2D(64,(4,4),2,activation = 'relu')
        self.l1 = tf.keras.layers.Conv2D(32,(3,3),(1,1),activation = 'relu',input_shape=(148,148,3))
        self.l2 = tf.keras.layers.Conv2D(16,(5,5),2,activation = 'relu')
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
        #print(prob)
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

tf.random.set_seed(336699)
agentoo7 = agent()
episode = 1000
ep_reward = []
total_avgr = []
dfrewards = []
game = GamePAI(1,'Connan',444,444,1,True,0)
for s in range(episode):
    done = False
    screen = game.screen
    array = pygame.surfarray.array3d(screen)
    array = array.swapaxes(0,1)
    array = array[:settings.mapHeigth,:settings.mapWidth]
    array =  np.expand_dims(array, axis=0)
    state = array/255
    total_reward = 0
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []
    steps = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        steps += 1
        action = agentoo7.act(state)
        next_state, reward = game.playerAction(action)
        done = game.gameOver(s)
        rewards.append(reward)
        states.append(state)
        #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
        actions.append(action)
        state = next_state
        total_reward += reward
        print(action,reward)
        if steps%1000 == 0:
            print(steps,total_reward,action,game.cave)

        if steps >= 1000 and game.cave < 2:
            noVideo = True
            if s% 100 == 0:
                game.__init__(1,'Connan',444,444,1,True,s)
                noVideo = False
            if noVideo:
                game.__init__(1,'Connan',444,444,1,True,s)
            gc.collect()
            done = True

        if steps%4 == 0:
            states, actions, discnt_rewards = agentoo7.preprocess1(states, actions, rewards, 1)
            al,cl = agentoo7.learn(states, actions, discnt_rewards) 
            states = []
            actions = []
            rewards = []
            #print(f"al{al}") 
            #print(f"cl{cl}")
    
        if done:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
            episodeStat = [s,total_reward,avg_reward]
            dfrewards.append(episodeStat)
dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
dfrewards.to_excel(r'D:\ekpa\diplomatiki\playerActor2CNNnStep.xlsx')
