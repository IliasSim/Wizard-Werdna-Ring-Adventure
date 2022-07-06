import queue
from wsgiref.util import setup_testing_defaults
from GamePAI_multi_agent import GamePAI_multi_agent
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
    
        


class Agent():
    '''agent is a class that creates the agent who play the game using the actor and critic network.
    Also contains functions for the training of the networks'''
    def __init__(self,agent_i):
        # self.gamma = gamma
        # self.initial_learning_rate = initial_learning_rate
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate=self.initial_learning_rate,
        # decay_steps=decay_steps,
        # decay_rate=0.9)
        
        self.agent_i = agent_i
        # print(self.global_critic)
        self.local_actor = actor()
        self.local_critic = critic()
        self.local_actor(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
        self.local_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
        # print(self.local_critic)                   
        # self.local_actor.set_weights(self.global_actor.get_weights())
        # self.local_critic.set_weights(self.global_critic.get_weights())
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.log_prob = None
        self.buffer_State = []
        self.buffer_playerStatus = []
        self.buffer_text = []
        self.buffer_reward = []
        self.buffer_size = h_step

    def act(self,state,playerstatus,gameText):
        '''This function use the actor NN in order to produce an action as an output'''
        prob = self.local_actor(state,playerstatus,gameText)
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

        state_nstep = []
        playerstatus_nstep = []
        gameText_nstep = []
        rewards_nstep = []
        actions_nstep = []
        state = []
        playerstatus = []
        gameText = []
        actions = []
        rewards = []
        for i in range(len(batch)):
            if (len(batch) - i) >= n_step:
                for z in range(n_step):
                    state_nstep.append(batch[i+z][0])
                    playerstatus_nstep.append(batch[i+z][1])
                    gameText_nstep.append(batch[i+z][2])
                    rewards_nstep.append(batch[i+z][3])
                    actions_nstep.append(batch[i+z][4])
            if (len(batch) - i) < n_step:
                for z in range(len(batch) - i):
                    state_nstep.append(batch[i+z][0])
                    playerstatus_nstep.append(batch[i+z][1])
                    gameText_nstep.append(batch[i+z][2])
                    rewards_nstep.append(batch[i+z][3])
                    actions_nstep.append(batch[i+z][4])
            state_nstep = np.array(state_nstep,dtype=np.float32)
            state_nstep =  np.squeeze(state_nstep,axis = 1)
            playerstatus_nstep = np.array(playerstatus_nstep,dtype=np.float32)
            playerstatus_nstep =  np.squeeze(playerstatus_nstep,axis = 1)
            gameText_nstep = np.array(gameText_nstep,dtype=np.float32)
            gameText_nstep =  np.squeeze(gameText_nstep,axis = 1)
            state.append(state_nstep)
            playerstatus.append(playerstatus_nstep)
            gameText.append(gameText_nstep)
            rewards.append(rewards_nstep)
            actions.append(actions_nstep)
            state_nstep = []
            playerstatus_nstep = []
            gameText_nstep = []
            rewards_nstep = []
            actions_nstep = []
               
        return  state,playerstatus,gameText,rewards,actions

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
        loss = -policy_loss - 0.0001 * entropy_loss
        return loss

    def learn(self, states,playerstatus,gameTexts, discnt_rewards,actions):
        '''This function is used for the training of the network, it also contain a code chunk for the depiction of the image used for the training. 
        For critic loss, we took a naive way by just taking the square of the temporal difference.'''
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.local_actor(states,playerstatus,gameTexts)
            v =  self.local_critic(states,playerstatus,gameTexts)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5*kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.local_actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.local_critic.trainable_variables)
        # self.a_opt.apply_gradients(zip(grads1, global_actor.trainable_variables))
        # self.c_opt.apply_gradients(zip(grads2, global_critic.trainable_variables))
        # self.local_actor.set_weights(global_actor.get_weights())
        # self.local_critic.set_weights(global_critic.get_weights())
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
        return a_loss, c_loss ,grads1 ,grads2 

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


def runner(total_episode,agent_i,queue_actor,queue_critic,grads_actor,grads_critic):
    # tf.random.set_seed(39999)
        global_actor = actor()
        global_critic = critic()
        global_actor(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
        global_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
        global_actor.set_weights(queue_actor)
        global_critic.set_weights(queue_critic)
        agent = Agent(agent_i)
        agent.local_actor.set_weights(queue_actor)
        agent.local_critic.set_weights(queue_critic)
        replay_memory = replay(memory_size)
        dir = 'D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\episodes_agent' + str(agent_i) + '.txt'
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
        game = GamePAI_multi_agent(1,'Connan',444,444,screenfactor,True,episodes_text,False,seeded,agent_i)
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
                if steps%truncate == 0 or done:
                    agent.local_actor.set_weights(global_actor.get_weights())
                    agent.local_critic.set_weights(global_critic.get_weights())
                    total_agent_actor_grads = [tf.zeros_like(this_var) for this_var in global_actor.trainable_variables]
                    total_agent_critic_grads = [tf.zeros_like(this_var) for this_var in global_critic.trainable_variables]
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
                action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
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
                        train_states,train_playerstatus,train_gametexts,train_rewards,train_actions= agent.preprocess1(replay_memory.replay_buffer)
                        discounted_rewards_all = []
                        for i in range(len(train_states)):
                            if len(train_states[i]) < n_step:
                                    reward_sum = 0.  # terminal
                            if len(train_states[i]) == n_step:
                                states_n_steps,playerstatus_n_steps,gametexts_n_steps = agent.preprocess2(train_states[i][n_step-1],train_playerstatus[i][n_step-1],train_gametexts[i][n_step-1])
                                reward_sum = agent.local_critic(states_n_steps,playerstatus_n_steps,gametexts_n_steps)
                                # Get discounted rewards
                            discounted_rewards = []
                            # if i == 0:
                                # print(agent_i,reward_sum)
                            for reward in train_rewards[i][::-1]:  # reverse buffer r
                                reward_sum = reward + gamma * reward_sum
                                discounted_rewards.append(reward_sum)
                            discounted_rewards.reverse()
                            discounted_rewards_all.append(discounted_rewards)
                            discounted_rewards = []
                        for i in range(len(train_states)):
                            al,cl,actor_grads,critic_grads = agent.learn(train_states[i],train_playerstatus[i],train_gametexts[i], discounted_rewards_all[i], train_actions[i])
                            print('agent ' +  str(agent_i) + ' Actor losses ' + str(al) + ' Critic Losses ' + str(cl) +' Discounted reward ' + str(discounted_rewards_all[i]) + ' for action ' + str(train_actions[i]))
                            total_agent_actor_grads = [(acum_grad+grad) for acum_grad, grad in zip(total_agent_actor_grads, actor_grads)]
                            total_agent_critic_grads = [(acum_grad+grad) for acum_grad, grad in zip(total_agent_critic_grads, critic_grads)]
                        grads_actor = [(acum_grad+grad) for acum_grad, grad in zip(grads_actor, total_agent_actor_grads)]
                        grads_critic = [(acum_grad+grad) for acum_grad, grad in zip(grads_critic, total_agent_critic_grads)]
                        agent.a_opt.apply_gradients(zip(grads_actor, global_actor.trainable_variables))
                        agent.c_opt.apply_gradients(zip(grads_critic, global_critic.trainable_variables))
                        queue_actor = global_actor.get_weights()
                        queue_critic = global_critic.get_weights()
                        
                        
                            

                if done:
                    total_episodes = 0
                    for key in total_episode.keys():
                        total_episodes = total_episodes + total_episode[key]
                    print('total episode' + str(total_episodes))
                    if total_episodes%100 == 0 and total_episodes !=0:
                        print('do i work')
                        global_actor.save_weights('.\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ actor_model2')
                        global_actor.save_weights('.\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ critic_model2')
                        f = open("D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ agent.txt",'w')
                        f.write(str(agent_i))
                        f.close()
                        f = open("D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ episodes.txt",'w')
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
    global_actor = actor()
    global_critic = critic()
    global_actor(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
    global_critic(tf.convert_to_tensor(np.random.random((input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((input3)), dtype=tf.float32))
    if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ actor_model2.data-00000-of-00001'):
        print("actor model is loaded")
        global_actor.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ actor_model2')
    if exists('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ critic_model2.data-00000-of-00001'):
        print("critic model is loaded")
        global_critic.load_weights('D:\ekpa\diplomatiki\Wizard-Werdna-Ring-Adventure\modelLSTM_Imp_15_12_21_nreplay_n_step_truncate_multiple_agents_1\ critic_model2')
    actor_vars = global_actor.trainable_variables
    critic_vars = global_critic.trainable_variables
    actor_accum_gradient = [tf.zeros_like(this_var) for this_var in actor_vars]
    critic_accum_gradient = [tf.zeros_like(this_var) for this_var in critic_vars]
    episode_dict = {}
    for i in range(mp.cpu_count()):
        episode_dict[str(i)] = 0
    total_episode = mp.Manager().dict(episode_dict)
    queue_actor = mp.Manager().list(global_actor.get_weights())
    queue_critic = mp.Manager().list(global_critic.get_weights())
    grads_actor = mp.Manager().list(actor_accum_gradient)
    grads_critic = mp.Manager().list(critic_accum_gradient)
    processes = []
    for i in range(mp.cpu_count()):
        worker = mp.Process(target = runner, args=(total_episode,i,queue_actor,queue_critic,grads_actor,grads_critic))
        processes.append(worker)
        worker.start()

    for process in processes:
        process.join()


    
