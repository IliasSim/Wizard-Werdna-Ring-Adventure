import queue
from wsgiref.util import setup_testing_defaults
from GamePAI import GamePAI
import pygame
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import gc
import tensorflow.keras.losses as kls
from os.path import exists
# import global_models
import sys
import multiprocessing as mp


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
seeded = False
input = (148,148,3)
record = False
batch_size = 12
truncate = 32
memory_size = truncate
gamma = 0.99
input1 = (1, 1, 148, 148, 3)
input2 = (1,1,10)
input3 = (1,1,125)

class actor_critic(tf.keras.Model):
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
        self.a = tf.keras.layers.Dense(8,activation ='softmax')
      

    def call(self,input_data,input_data1,input_data2):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        print(x.shape)
        x = self.l5(x)
        y = self.l7(input_data1)
        z = self.l9(input_data2)
        h = self.conc1((y,z))
        g = self.conc2((h,x))
        v = self.v(g)
        a = self.a(g)
        return a,v

class Agent():
    '''agent is a class that creates the agent who play the game using the actor and critic network.
    Also contains functions for the training of the networks'''
    def __init__(self,agent_i):
        
        self.agent_i = agent_i
        # print(self.global_critic)
        self.local_actor_critic = actor_critic()
        self.local_actor_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
        
        self.a_c_opt = tf.keras.optimizers.Adam(learning_rate=0.0025)
        self.log_prob = None
        self.buffer_State = []
        self.buffer_playerStatus = []
        self.buffer_text = []
        self.buffer_reward = []
        self.buffer_size = h_step

    def act(self,state,playerstatus,gameText):
        '''This function use the actor NN in order to produce an action as an output'''
        prob,_ = self.local_actor_critic(state,playerstatus,gameText)
        del _
        print(self.agent_i,prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def preprocess0(self, state,playerstatus,gameText):
        '''This function cretaes the history of observation for the input to the NN. Also calculate
        the valeu of the last observation based on past and current rewards.'''
       
        if len(self.buffer_State) >= h_step:
            del self.buffer_State[0]
        self.buffer_State.append(state)
        state = np.array(self.buffer_State,dtype=np.float32)
        state = np.expand_dims(state, axis=0)

        if len(self.buffer_playerStatus) >= h_step:
            del self.buffer_playerStatus[0]
        self.buffer_playerStatus.append(playerstatus)
        playerstatus = np.array(self.buffer_playerStatus, dtype=np.float32)
        playerstatus = np.expand_dims(playerstatus, axis=0)

        if len(self.buffer_text) >= h_step:
            del self.buffer_text[0]
        self.buffer_text.append(gameText)
        gameText = np.array(self.buffer_text, dtype=np.float32)
        gameText = np.expand_dims(gameText, axis=0)
        # print(state.shape)
        return state,playerstatus,gameText

    def preprocess1(self,batch):
        '''This function take four memory registrations and modified them in order to be used for the training of the network'''
        state = []
        playerstatus = []
        gameText = []
        actions = []
        rewards = []
        rewards_nstep = []
        for i in range(len(batch)):
            state.append(batch[i][0])
            playerstatus.append(batch[i][1])
            gameText.append(batch[i][2])
            actions.append(batch[i][4])
        for i in range(len(batch)):
            if (len(batch) - i) >= n_step:
                for z in range(n_step):
                    rewards_nstep.append(batch[i+z][3])
            if (len(batch) - i) < n_step:
                for z in range(len(batch) - i):
                    rewards_nstep.append(batch[i+z][3])
            rewards.append(rewards_nstep)
            rewards_nstep= []
        discounted_rewards = []
        # print(rewards)
        for i in range(len(rewards)):
            if len(rewards[i]) < n_step:
                reward_sum = tf.zeros(shape = (1,1),dtype=tf.dtypes.float32)  # terminal
            if len(rewards[i]) == n_step:
                _,reward_sum = self.local_actor_critic(state[i + n_step-1],playerstatus[i + n_step-1],gameText[i + n_step-1])
                del _
            for reward in rewards[i][::-1]:  # reverse buffer r
                reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        # print(discounted_rewards)

        # discounted_rewards.reverse()
        state = np.array(state,dtype=np.float32)
        state =  np.squeeze(state,axis = 1)
        playerstatus = np.array(playerstatus,dtype=np.float32)
        playerstatus =  np.squeeze(playerstatus,axis = 1)
        gameText = np.array(gameText,dtype=np.float32)
        gameText =  np.squeeze(gameText,axis = 1)
        return  state,playerstatus,gameText,discounted_rewards,actions

    def preprocess2(self,train_state,train_playerstatus,train_gameText):
        train_state = np.expand_dims(train_state,axis=0)
        train_playerstatus = np.expand_dims(train_playerstatus,axis = 0)
        train_gameText = np.expand_dims(train_gameText,axis = 0)
        return train_state,train_playerstatus,train_gameText


    def actor_loss(self, probs, actions, td):
        '''This function calculate actor NN losses. Which is negative of Log probability of action taken multiplied 
        by temporal difference used in q learning.'''
        probability = []
        log_probability= []
        for pb,a in zip(probs,actions):
            dist = tfp.distributions.Categorical(probs, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)
        p_loss= []
        e_loss = []
        td = td.numpy()
        
        for pb,t,lpb in zip(probability, td, log_probability):
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

    def learn(self, states,playerstatus,gameTexts, discnt_rewards,actions):
        '''This function is used for the training of the network, it also contain a code chunk for the depiction of the image used for the training. 
        For critic loss, we took a naive way by just taking the square of the temporal difference.'''
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape:
            p,v = self.local_actor_critic(states,playerstatus,gameTexts)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(discnt_rewards, v)
            loss = a_loss + c_loss
        grads_a_c = tape.gradient(loss, self.local_actor_critic.trainable_variables)
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
        return loss,grads_a_c

class replay():
    '''This class used as a buffer for the replay memory. Batches of 4 memory units are extracted from the memory and feeded at the network.
    Each memory unit contains 4 observations 3 for the history input at the lstms and one last observation. For the last observation also
     included the value it posses and the action which follow up based on that observation.'''
    def __init__(self,memory_size):
        self.memory_size = memory_size
        self.replay_buffer = []


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


def runner(total_episode,agent_i,manager_actor_critic,grads_actor_critic,barrier,lock):
    # tf.random.set_seed(39999)
        global_actor_critic = actor_critic()
        global_actor_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
        global_actor_critic.set_weights(manager_actor_critic)
        agent = Agent(agent_i)
        agent.local_actor_critic.set_weights(manager_actor_critic)
        replay_memory = replay(memory_size)
        dir = 'D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_5_common\episodes_agent' + str(agent_i) + '.txt'
        if not exists(dir):
            f = open(dir,'w+')
            f.write('1')
        f = open(dir,'r')
        episodes_text = int(f.read())
        f.close()
        episode = 10000
        ep_reward = []
        total_avgr = []
        dfrewards = []
        # s = episodes_text
        game = GamePAI(1,'Connan',444,444,screenfactor,True,episodes_text,False,seeded,agent_i)
        game_No = episodes_text
        for s in range(episodes_text,episode):
            f = open(dir,'w')
            f.write(str(s))
            f.close()
            if len(total_episode) == 8:
                del total_episode[str(agent_i)]
            total_episode[str(agent_i)] = s
            game_No = game_No + 1
            done = False
            state,playerStatus, gameText = game.initialGameState()
            rewards = []
            total_reward = 0
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
                state,playerStatus,gameText = agent.preprocess0(state,playerStatus,gameText)
                buffer_state,buffer_playerStatus,buffer_gameText = state,playerStatus,gameText
                action = agent.act(state,playerStatus,gameText)
                next_state, next_playerStatus, next_gameText,reward,done  = game.playerAction(action)
                rewards.append(reward)
                replay_memory.create_memory(buffer_state,buffer_playerStatus,buffer_gameText,reward,action)
                action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'attack',7:'pick'}
                print(agent_i,action_name[action],reward,game.cave,steps,s)
                total_reward += reward
                state = next_state
                playerStatus = next_playerStatus
                gameText = next_gameText
                if done:
                    game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,agent_i)

                if steps%1000 == 0:
                    print(steps,total_reward,action_name[action],game.cave)

                if s <= 2000: 
                    if steps >= 500 and game.cave < 2:
                        noVideo = True
                        if s% 100 == 0:
                            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,agent_i)
                            noVideo = False
                        if noVideo:
                            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,agent_i)
                        gc.collect()
                        print(s,total_reward,game.cave)
                        done = True
                if s > 2000: 
                    if steps >= 5000 and game.cave < 2:
                        noVideo = True
                        if s% 100 == 0:
                            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,agent_i)
                            noVideo = False
                        if noVideo:
                            game.__init__(1,'Connan',444,444,screenfactor,True,game_No,False,seeded,agent_i)
                        gc.collect()
                        print(s,total_reward,game.cave)
                        done = True
                # print(agent_i,steps,steps%truncate == 0  or done)

                if steps > truncate + h_step - 1:
                    if steps%truncate == 0 or done:
                        train_states,train_playerstatus,train_gametexts,discounted_rewards,train_actions= agent.preprocess1(replay_memory.replay_buffer)
                        # for i in range(len(train_states)):
                            # train_states_e,train_playerstatus_e,train_gametexts_e = agent.preprocess2(train_states[i],train_playerstatus[i],train_gametexts[i])
                        print('i work')
                        loss,actor_critic_grads = agent.learn(train_states,train_playerstatus,train_gametexts,discounted_rewards,train_actions)
                        print('agent ' +  str(agent_i) + ' Actor-Critic Losses ' + str(loss))
                        lock.acquire()
                        grads_actor_critic = [(acum_grad+grad) for acum_grad, grad in zip(grads_actor_critic, actor_critic_grads)]
                        lock.release()
                        # grads_actor_critic = [(acum_grad+grad) for acum_grad, grad in zip(grads_actor_critic, total_agent_actor_critic_grads)]
                        b = barrier.wait()
                        if b ==0:
                            print('I do the zeroisation' + str(agent_i))
                            agent.a_c_opt.apply_gradients(zip(grads_actor_critic, global_actor_critic.trainable_variables))
                            manager_actor_critic = global_actor_critic.get_weights()
                            print(len(manager_actor_critic))
                            grads_actor_critic = [tf.zeros_like(this_var) for this_var in global_actor_critic.trainable_variables]
                        global_actor_critic.set_weights(manager_actor_critic)
                        agent.local_actor_critic.set_weights(manager_actor_critic)
                        
                if done:
                    total_episodes = 0
                    for key in total_episode.keys():
                        total_episodes = total_episodes + total_episode[key]
                    print('total episode' + str(total_episodes))
                    if total_episodes%100 == 0 and total_episodes !=0:
                        print('do i work')
                        global_actor_critic.save_weights('.\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_5_common\ actor_critic_model2')
                        f = open("D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_5_common\ agent.txt",'w')
                        f.write(str(agent_i))
                        f.close()
                        f = open("D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_5_common\ episodes.txt",'w')
                        f.write(str(total_episodes))
                        f.close()
                        ep_reward.append(total_reward)
                        avg_reward = np.mean(ep_reward[-100:])
                        total_avgr.append(avg_reward)
                        print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
                        episodeStat = [s,total_reward,avg_reward]
                        dfrewards.append(episodeStat)
        dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
        dfrewards.to_excel(r'D:\ekpa\diplomatiki\playerActor2CNNnStepwithtext.xlsx')
        
        

if __name__ == '__main__':
    global_actor_critic = actor_critic()
    global_actor_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
    if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_5_common\ actor_critic_model2.data-00000-of-00001'):
        print("actor critic model is loaded")
        global_actor_critic.load_weights('.\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_5_common\ actor_critic_model2')
    actor_critic_vars = global_actor_critic.trainable_variables
    actor_critic_accum_gradient = [tf.zeros_like(this_var) for this_var in actor_critic_vars]
    episode_dict = {}
    for i in range(mp.cpu_count()):
        episode_dict[str(i)] = 0
    total_episode = mp.Manager().dict(episode_dict)
    manager_actor_critic = mp.Manager().list(global_actor_critic.get_weights())
    grads_actor_critic = mp.Manager().list(actor_critic_accum_gradient)
    barrier = mp.Barrier(mp.cpu_count())
    lock = mp.Lock()
    processes = []
    for i in range(mp.cpu_count()):
        worker = mp.Process(target = runner, args=(total_episode,i,manager_actor_critic,grads_actor_critic,barrier,lock))
        processes.append(worker)
        worker.start()

    for process in processes:
        process.join()


    
