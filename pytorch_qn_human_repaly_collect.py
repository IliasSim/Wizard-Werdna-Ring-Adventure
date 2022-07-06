from GamePAI_pytorch_dqn_weapon_posses import GamePAI
import matplotlib.pyplot as plt
import sys
import pygame
import numpy
import random
import pickle
import os
import numpy as np
from os.path import exists
screenfactor = 1
seeded = False
record = False
truncate = 32
memory_size = 350000
class Sequence_Buffer:
    def __init__(self):
        self.buffer_size = truncate
        self.buffer = []
        self.epsilon = 1

    def eraseMemory(self):
        self.buffer = []

    def WriteMemory(self,last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action):
        if len(self.buffer) >= memory_size:
            del self.buffer[0]
        self.buffer.append([last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action])

    def TakeSequence(self,batch_size):
        batch_start_boolean = True
        if len(self.buffer) >= batch_size:
            while batch_start_boolean:
                batch_start = random.randint(0,len(self.buffer)-1)
                if len(self.buffer) - batch_start >= batch_size:
                    batch_start_boolean = False
            sequence = self.buffer[batch_start:(batch_start+batch_size)]   
        return sequence

def Human_playing():
    buffer = Sequence_Buffer()
    game_cont = True
    agent = 0
    g = 0
    if exists('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt'):
        size = os.path.getsize('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt') 
    else: 
        with open('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'x') as f:
            f.write('')
            size = 0
    if size >0:
        with open ('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'rb') as fp:
            buffer.buffer = pickle.load(fp)
    while game_cont:   
        g += 1
        game = GamePAI(1,'Connan',444,444,screenfactor,True,g,False,seeded,agent)
        last_state,last_playerStatus, last_gameText = game.initialGameState()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    with open('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'wb') as fp:
                        pickle.dump(buffer.buffer, fp)
                    game_cont = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action = 3
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(3)
                    if event.key == pygame.K_s:
                        action = 2
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(2)
                    if event.key == pygame.K_d:
                        action = 1
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(1)
                    if event.key == pygame.K_w:
                        action = 0
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(0)
                    if event.key == pygame.K_r:
                        action = 4
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(4)
                    if event.key == pygame.K_h:
                        action = 5
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(5)
                    if event.key == pygame.K_m:
                        action = 8
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(8)
                    if event.key == pygame.K_SPACE:
                        action = 6
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(6)
                    if event.key == pygame.K_p:
                        action = 7
                        next_state, next_playerStatus, next_gameText,reward,done = game.playerAction(7)
                    if done == True:
                        with open('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'wb') as fp:
                            pickle.dump(buffer.buffer, fp)
                        pygame.quit()
                    buffer.WriteMemory(last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action)
                    last_state,last_playerStatus,last_gameText = next_state, next_playerStatus, next_gameText
                    print(len(buffer.buffer))
                    

if __name__ == '__main__':
    Human_playing()
