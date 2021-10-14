import pygame
from GamePAI import GamePAI
import random
import matplotlib.pyplot as plt
import GameSettings as settings
import numpy as np
import time
import pandas as pd

start_time = time. time()
#seed = 42
#np.random.seed(seed)
game = GamePAI(1,'Connan',444,444,1,False,0)
#game.seed(seed)
#game.makeMap(game.cave)
#print(settings.xtile)

reward = 0
total_reward = []
dfrewards = []
for i in range(1000):
    done = False
    episode_reward = 0
    #total_reward = []
    while not done:
        x = random.random()
        #print(type(x))
        y = 1/9
        #print(type(3*y))
        if x < y:
            #print('up')
            screen,reward = game.playerAction(0)
        if x >= y and x < 2*y:
            #print('right')
            screen,reward =game.playerAction(1)
        if x >= 2*y and x < 3*y:
            #print('down')
            screen,reward =game.playerAction(2)
        if x >= 3*y and x < 4*y:
            #print('left')
            screen,reward =game.playerAction(3)
        if x >=4*y and x <5*y:
            #print('rest')
            screen,reward =game.playerAction(4)
        if x >= 5*y and x <6*y:
            #print('hp')
            screen,reward =game.playerAction(5)
        if x >= 6*y and x < 7*y:
            #print('mp')
            screen,reward =game.playerAction(6)
        if x >= 7*y and x < 8*y:
            #print("attack")
            screen,reward =game.playerAction(7)
        if x >= 8*y and x < 9*y:
            #print('pick')
            screen,reward =game.playerAction(8)
        done = game.gameOver(i)
        episode_reward += reward
        if done == True:
            total_reward.append(episode_reward)
            mean_reward = np.mean(total_reward[-100:])
            episodeStat = [i,episode_reward,mean_reward]
            dfrewards.append(episodeStat)
            print(game.player.name,'is dead. Game Over ' + str(i) + ' episode rewards ' + str(episode_reward) + ' mean reward:' + str(mean_reward))
    
    #array = pygame.surfarray.array3d(screen)
    #print(array.shape)
    #array = screen[0]
    #array = array*255
    #array = array.swapaxes(0,1)
    #array = array[:settings.mapHeigth,:settings.mapWidth]
    #array =  np.expand_dims(array, axis=0)
    #game.gameOver()
dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
#print(dfrewards)
dfrewards.to_excel(r'D:\ekpa\diplomatiki\lucky_agent_rewards4nStepsNormalize_to_one.xlsx')
print("--- %s seconds ---" % (time. time() - start_time))
#array = screen[0:game.settingmapHeigth(),0:game.settingmapWidth()]
#fig = plt.figure()
#plt.imshow(array)
#plt.title("Plot 2D array")
#plt.show()
    
