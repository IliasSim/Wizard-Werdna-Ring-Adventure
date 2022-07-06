import multiprocessing
from GamePAI_multi_agent import GamePAI_multi_agent
# import pygame
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
import sys
import random
from multiprocessing import Process, Manager, Barrier, Queue
import threading
import args


class ActorCritic(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv3D(args.featuresCNN1,(1,args.CNN1Shape,args.CNN1Shape),(1,args.CNN1Step,args.CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv3D(args.featuresCNN2,(1,args.CNN2Shape,args.CNN2Shape),(1,args.CNN2Step,args.CNN2Step),activation = 'relu')
        self.l3 = tf.keras.layers.Conv3D(args.featuresCNN3,(1,args.CNN3Shape,args.CNN3Shape),(1,args.CNN3Step,args.CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((-1,2704))
        self.l5 = tf.keras.layers.LSTM(args.denseLayerN)
        self.l7 = tf.keras.layers.LSTM(args.denseLayerNL_21)
        self.l9 = tf.keras.layers.LSTM(args.denseLayerNL_31)
        self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        self.v = tf.keras.layers.Dense(1)
        self.a = tf.keras.layers.Dense(9)
      

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
        a = self.a(g)
        return a,v


class RandomAgent:
    """Random agent plays the game using random actions for the game
    Arguments:
        max_episode: Maximum number of episodes to run agent for
    """
    def __init__(self,max_episode):
        self.max_episode = max_episode
    
    def run(self):
        game = GamePAI_multi_agent(1,'Connan',444,444,1,True,0,False,False,0)
        steps = 0
        reward = 0
        total_reward = 0
        dfrewards = []
        for i in range(1000):
            done = False
            episode_reward = 0
        while not done:
            steps += 1
            # for event in pygame.event.get():
                # if event.type == pygame.QUIT:
                    # pygame.quit()
                    # sys.exit()
            x = random.randrange(9)
            state, playerstatus, textArray, reward, done= game.playerAction(x)
            del state, playerstatus, textArray
            episode_reward += reward
            if steps >= 500:
                done = True
            if done:
                steps = 0
                game.__init__(1,'Connan',444,444,3,True,i,False,False,0)
                total_reward += episode_reward
                mean_reward = total_reward/(i+1)
                episodeStat = [i,episode_reward,mean_reward]
                dfrewards.append(episodeStat)
                print(game.player.name,'is dead. Game Over ' + str(i) + ' episode rewards ' + str(episode_reward) + ' mean reward:' + str(mean_reward))
                episode_reward = 0
        dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
        dfrewards.to_excel(r'D:\ekpa\diplomatiki\random_5-2-22.xlsx')

class Memory():
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

    def clear(self):
        self.replay_buffer = []

class Worker(threading.Thread):
    global_episode = 0
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()
    def __init__(self,global_model,opt,result_queue,idx,save_dir = '/tmp'):
        super(Worker,self).__init__()
        self.result_queue = result_queue
        self.global_model = global_model
        self.buffer_State = []
        self.buffer_playerStatus = []
        self.buffer_text = []
        self.opt = opt
        self.local_model = ActorCritic()
        self.worker_idx = idx
        self.game = GamePAI_multi_agent(1,'Connan',444,444,args.screenfactor,True,0,False,args.seeded,self.worker_idx)
        print(self.game)
        self.save_dir = save_dir
        self.ep_loss = 0

    

    def preprocess0(self, state,playerstatus,gameText):
        '''This function cretaes the history of observation for the input to the NN. Also calculate
        the valeu of the last observation based on past and current rewards.'''
       
        if len(self.buffer_State) >= args.h_step:
            self.buffer_State = []
        self.buffer_State.append(state)
        state = np.array(self.buffer_State,dtype=np.float32)
        state = np.expand_dims(state, axis=0)

        if len(self.buffer_playerStatus) >= args.h_step:
            self.buffer_playerStatus = []
        self.buffer_playerStatus.append(playerstatus)
        playerstatus = np.array(self.buffer_playerStatus, dtype=np.float32)
        playerstatus = np.expand_dims(playerstatus, axis=0)

        if len(self.buffer_text) >= args.h_step:
            self.buffer_text = []
        self.buffer_text.append(gameText)
        gameText = np.array(self.buffer_text, dtype=np.float32)
        gameText = np.expand_dims(gameText, axis=0)
        # print(state.shape)
        return state,playerstatus,gameText
   
    
    def run(self):
        
        total_step = 1
        episode = 0
        mem = Memory(args.memory_size)
        while Worker.global_episode < args.max_eps:
            # for event in pygame.event.get():
                # if event.type == pygame.QUIT:
                    # pygame.quit()
                    # sys.exit()
                # if event.type == pygame.KEYDOWN:
                    # if event.key == pygame.K_r:
                        # print('The record starts')
                       #  record = True
                     #if event.key == pygame.K_u:
                        # print('The record stops')
                        # record = False
            state,playerStatus, gameText = self.game.initialGameState()
            ep_reward = 0.
            episode += 1
            total_reward = 0
            ep_steps = 0
            self.ep_loss = 0
            time_count = 0
            done = False
            while not done:
                state,playerStatus, gameText = self.preprocess0(state,playerStatus,gameText)
                buffer_state,buffer_playerStatus,buffer_gameText = state,playerStatus,gameText
                logits, _ = self.local_model(state,playerStatus, gameText)
                probs = tf.nn.softmax(logits)
                print(probs)
                action = np.random.choice(9,p = probs.numpy()[0])
                next_state, next_playerStatus, next_gameText,reward,done  = self.game.playerAction(action)
                state,playerStatus, gameText = next_state, next_playerStatus, next_gameText
                mem.create_memory(buffer_state,buffer_playerStatus,buffer_gameText,reward,action)
                ep_reward += reward
                action_name = {0:'up',1:'right',2:'down',3:'left',4:'rest',5:'hp',6:'mp',7:'attack',8:'pick'}
                print(action_name[action],reward,self.game.cave,ep_steps,time_count,buffer_state.shape)
                if done:
                    self.game.__init__(1,'Connan',444,444,args.screenfactor,True,episode,False,args.seeded,self.worker_idx)

                if ep_steps%1000 == 0:
                    print(ep_steps,total_reward,action_name[action],self.game.cave)

                if episode <= 2000: 
                    if ep_steps >= 500 and self.game.cave < 2:
                        noVideo = True
                    if time_count% 100 == 0:
                        self.game.__init__(1,'Connan',444,444,args.screenfactor,True,episode,False,args.seeded,self.worker_idx)
                        noVideo = False
                    if noVideo:
                        self.game.__init__(1,'Connan',444,444,args.screenfactor,True,episode,False,args.seeded,self.worker_idx)
                    gc.collect()
                    print(time_count,total_reward,self.game.cave)
                    done = True
                if episode > 2000: 
                    if ep_steps >= 5000 and self.game.cave < 2:
                        noVideo = True
                    if time_count% 100 == 0:
                        self.game.__init__(1,'Connan',444,444,args.screenfactor,True,episode,False,args.seeded,self.worker_idx)
                        noVideo = False
                    if noVideo:
                        self.game.__init__(1,'Connan',444,444,args.screenfactor,True,episode,False,args.seeded,self.worker_idx)
                    gc.collect()
                    print(time_count,total_reward,self.game.cave)
                    done = True
                if time_count >= args.memory_size:
                    if time_count == args.n_step or done:
                        # Calculate gradient wrt to local model. We do so by tracking the
                        # variables involved in computing the loss by using tf.GradientTape
                        with tf.GradientTape() as tape:
                            total_loss = self.compute_loss(done,
                                           mem,
                                           args.gamma)
                        self.ep_loss += total_loss
                        # Calculate local gradients
                        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                        # Push local gradients to global model
                        self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))
                        # Update local model with new weights
                        self.local_model.set_weights(self.global_model.get_weights())
                        time_count = 0
                        if done:  # done and print information
                            # Worker.global_moving_average_reward = \
                            # record(Worker.global_episode, ep_reward, self.worker_idx,
                            # Worker.global_moving_average_reward, self.result_queue,
                            # self.ep_loss, ep_steps)
                            # We must use a lock to save our model and to print to prevent data races.
                            if ep_reward > Worker.best_score:
                                with Worker.save_lock:
                                    print("Saving best model to {}, "
                                        "episode score: {}".format(self.save_dir, ep_reward))
                                    self.global_model.save_weights(
                                        os.path.join(self.save_dir,
                                                'model_{}.h5'.format(self.game_name))
                            )
                            Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1
                time_count += 1
                total_step += 1
            self.result_queue.put(None)

    def compute_loss(self,
                   done,
                   memory,
                   gamma=0.99):
        x_1,x_2,x_3,x_4 = memory.replay_buffer[-4], memory.replay_buffer[-3], memory.replay_buffer[-2], memory.replay_buffer[-1]
        train_state_all = [x_1[0],x_2[0],x_3[0],x_4[0]]
        train_playerstatus_all = [x_1[1],x_2[1],x_3[1],x_4[1]]
        train_gametex_all = [x_1[2],x_2[2],x_3[2],x_4[2]]
        train_rewards = [x_1[3],x_2[3],x_3[3],x_4[3]]
        train_actions = [x_1[4],x_2[4],x_3[4],x_4[4]]
        if done:
            reward_sum = 0.  # terminal
        else:

            reward_sum = self.local_model(train_state_all[3],train_playerstatus_all[3],train_gametex_all[3])[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in train_rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(train_state_all[:2],train_playerstatus_all[:2],train_gametex_all[:2])
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                            dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(train_actions, 9, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)

        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
                                                             logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

class MasterAgent():
    def __init__(self):
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.opt = tf.compat.v1.train.AdamOptimizer(args.learning_rate, use_locking = True)
        self.global_model = ActorCritic()
        self.global_model(tf.convert_to_tensor(np.random.random((args.input1)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((args.input2)), dtype=tf.float32),
                            tf.convert_to_tensor(np.random.random((args.input3)), dtype=tf.float32))

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(1000)
            random_agent.run()
            return
        res_queue = Queue()
        game_list = []

        workers = [Worker(self.global_model,
                            self.opt,res_queue,i,self.save_dir)for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):

            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                             '{} Moving Average.png'.format(self.game_name)))
        plt.show()

if __name__ == '__main__':
    master = MasterAgent()
    master.train()


            