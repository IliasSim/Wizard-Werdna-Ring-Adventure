import pygame
from GamePAI import GamePAI
import random
import matplotlib.pyplot as plt
import GameSettings as settings
import pygame
import numpy as np
import sys
#import pandas as pd

#start_time = time. time()
game = GamePAI(1,'Connan',444,444,3,True,0)
#print(settings.xtile)
#screen = game.screen
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            reward = 0
        total_reward = 0
        done = False
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
        #game.gameOver(0)
        screen = screen[0]
        printArray(screen)
        #if done == True:
        #total_reward += reward
        #mean_reward = total_reward/(i+1)
        #episodeStat = [i,reward,mean_reward]
        #dfrewards.append(episodeStat)

    #print(game.player.name,'is dead. Game Over ' + str(i) + ' episode rewards ' + str(reward) + ' mean reward:' + str(mean_reward))
    
    #array = pygame.surfarray.array3d(screen)
    #print(array.shape)
    def printArray(screen):
        array = screen[0]
        print(array.shape)
        array = array*255
    #array = array.swapaxes(0,1)
        array = array[:settings.mapHeigth,:settings.mapWidth]
    #array =  np.expand_dims(array, axis=0)
    #game.gameOver()
#dfrewards = pd.DataFrame(dfrewards, columns=['episode', 'rewards', 'mean rewards'])
#print(dfrewards)
#dfrewards.to_excel(r'D:\ekpa\diplomatiki\lucky_agent_rewards4.xlsx')
#print("--- %s seconds ---" % (time. time() - start_time))
        array = array[0:game.settingmapHeigth(),0:game.settingmapWidth()]
        fig = plt.figure()
        plt.imshow(array)
        plt.title("Plot 2D array")
        plt.show()
    
