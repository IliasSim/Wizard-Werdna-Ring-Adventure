import sys
import GameSettings as settings
from GameMap import GameMap
import pygame
from GameMapGraphics import GameMapGraphics
from Games_items import HelthPotion,ManaPotion,Staff,Sword,Werdna_Ring
from Warrior import Warrior
from Wizard import Wizard
from Enemy import Enemy
import GameEnum
import random
from Tile import Tile
import numpy as np
import gc
eps = np.finfo(np.float32).eps.item()

class GamePAI():
    '''GamePAi is a class that creates an instance of the game and initialize the basic features of the game.'''
    def __init__(self,playerType,playerName,xPixel,yPixel,screenFactor,depict,game):
        self.xPixel = xPixel
        self.yPixel = yPixel
        self.screenFactor = screenFactor
        settings.screenFactor = screenFactor
        settings.mapWidth = int(self.xPixel/3)*screenFactor
        settings.mapHeigth = int(self.yPixel/3)*screenFactor
        settings.xtile = int(settings.mapWidth/(4*screenFactor))
        settings.ytile = int(settings.mapHeigth/(4*screenFactor))
        settings.tiles = [[0]*settings.ytile for i in range(settings.xtile)]
        for y in range(settings.ytile):
            for x in range(settings.xtile):
                settings.tiles[x][y] = Tile(x,y)
        self.depict = depict
        self.steps = 0
        self.reward = 0
        #random.seed(seed)
        if depict == True:
            self.screen = pygame.display.set_mode((settings.mapWidth+150, settings.mapHeigth+70))
        if depict == False:
            self.screen = pygame.Surface((settings.mapWidth+150, settings.mapHeigth+70))
        self.DrawMap = GameMapGraphics(self.screen)
        self.map = GameMap()
        self.cave = 0
        self.playerType = playerType
        self.playerName = playerName
        self.total_reward = []
        settings.enemies = []
        settings.game_text = []
        pygame.init()
        if playerType == 1:
            self.player = Warrior()
        if playerType == 2:
            self.player = Wizard()
        self.player.name = playerName          
        text = 'Welcome to the Wizard Werdna Ring adventure. Try to find the ring and win the Game.'
        settings.addGameText(text)
        self.printText()
        self.makeMap(self.cave)
        self.printPlayerStatus()
        if depict == True:
            pygame.display.set_caption("Wizard Werdna Ring "+str(game+1))
            pygame.display.flip()

    def afterMoveDepiction(self):
        '''This function refreshes the information on the game screen'''
        self.printText()
        self.printPlayerStatus()
        self.drawMap() 
        

    def gameOver(self,s):
        '''Checks if the player is dead and reinitilize the game creating a new instance of the GamePAI class.'''
        self.done = False
        if self.player.hitPoints <= 0:
            self.done = True
            #self.reward -= 1000
            settings.gameCount = settings.gameCount + 1
            #print(self.player.name,'is dead. Game Over ' + str(settings.gameCount) + ' episode rewards ' + str(self.reward))
            pygame.quit()
            #sys.exit()
            if s % 100 == 0:
                self.__init__(self.playerType,self.playerName,self.xPixel,self.yPixel,self.screenFactor,True,s)
            else:
                self.__init__(self.playerType,self.playerName,self.xPixel,self.yPixel,self.screenFactor,True,s)
            gc.collect()
        return self.done

    def drawMap(self):
        '''This function draws the map of the game'''
        self.DrawMap.drawMap()
        self.DrawMap.drawItem()
        self.DrawMap.enemyDepiction()
        self.DrawMap.drawPlayer(self.player.currentPositionX, self.player.currentPositionY)
        #FramePerSec = pygame.time.Clock()
        #FramePerSec.tick(20)
        if self.depict == True:
            pygame.display.update()

    def makeMap(self,cave):
        '''Creates the map of the game.'''
        precentage = 0
        makemap = True 
        while makemap:
            precentage = random.random()
            if 0.35 < precentage <= 0.55:
                makemap = False
        if cave >= 0:
            self.map.refreshTilesSettings()
        self.map.MakeMAp(precentage, self.player, cave)
        self.player.playerVisibility(self.player.currentPositionX, self.player.currentPositionY)

    def countHealthPotion(self):
        '''Counts the health potions player posses.'''
        count = 0
        if self.player.inventory != []:
            for item in self.player.inventory:
                if isinstance(item, HelthPotion):
                    count = count + 1
        return count
 
    def countManaPotion(self):
        '''Counts the health potions player posses.'''
        count = 0
        if self.player.inventory != []:
            for item in self.player.inventory:
                if isinstance(item,ManaPotion):
                    count = count + 1
        return count

    def printPlayerStatus(self):
        '''Prints the player name and status'''
        font = pygame.font.Font("freesansbold.ttf", 12)
        if isinstance(self.player, Warrior):
            text = font.render('Warrior', True, (255,0,0), (0,0,0))
        if isinstance(self.player, Wizard):
            text = font.render('Wizard', True, (255,0,0), (0,0,0))
        text1 = font.render(self.player.name, True, (255,0,0), (0,0,0))
        text2 = font.render("HP" + str(self.player.hitPoints) + " /" + str(self.player.maxHitPoints), True, (255,0,0), (0,0,0)) 
        if isinstance(self.player, Warrior):
            text25 = font.render("MP 0/0" , True, (255,0,0), (0,0,0))    
        if isinstance(self.player, Wizard):
            text25 = font.render("MP" + str(self.player.manaPoints) + "/" + str(self.player.maxManaPoints) , True, (255,0,0), (0,0,0))  
        if self.player.weapon == None:
            text3 = font.render("Weapon: None " , True, (255,0,0), (0,0,0))
        else:
            text3 = font.render("Weapon: " + self.player.weapon.name , True, (255,0,0), (0,0,0))
        if isinstance(self.player, Warrior):
            text4 = font.render('Strength: ' + str(self.player.baseStrength), True, (255,0,0), (0,0,0))
        if isinstance(self.player, Wizard):
            text4 = font.render('Intelligence: ' + str(self.player.getBaseIntelligence()), True, (255,0,0), (0,0,0))
        text5 = font.render('Damage: ' + str(self.player.getAttackDamage()), True, (255,0,0), (0,0,0))
        text6 = font.render('Potions: ' + str(self.countHealthPotion())+ "(H) /" + str(self.countManaPotion()) + "(M)", True, (255,0,0), (0,0,0))
        text7 = font.render('Level: ' + str(self.player.getLevel()), True, (255,0,0), (0,0,0))
        text8 = font.render('XP: ' + str(self.player.experiencePoints), True, (255,0,0), (0,0,0)) 
        self.screen.blit(text, (settings.mapWidth,8))
        self.screen.blit(text1, (settings.mapWidth,21))
        self.screen.blit(text2, (settings.mapWidth,34))
        self.screen.blit(text25, (settings.mapWidth,47))
        self.screen.blit(text3, (settings.mapWidth,60))
        self.screen.blit(text4, (settings.mapWidth,73))
        self.screen.blit(text5, (settings.mapWidth,86))
        self.screen.blit(text6, (settings.mapWidth,99))
        self.screen.blit(text7, (settings.mapWidth,112))
        self.screen.blit(text8, (settings.mapWidth,125))

    def printText(self):
        '''Prints the game Log on the screen'''
        font = pygame.font.Font('freesansbold.ttf', 12)
        self.screen.fill(pygame.Color("black"))
        for i in range(len(settings.game_text)):
            text = font.render(settings.game_text[i], True, (255,0,0), (0,0,0))
            self.screen.blit(text,(0,settings.mapHeigth+(13*i)))
    
    def enemyMove(self):
        '''Determines the movement of the enemy after the movement of the player.'''
        if settings.enemies != []:
            for enemy in settings.enemies:
                #if enemy.minDistance(self.player,enemy.enemyCurrentPossitionX,enemy.enemyCurrentPossitionY) == 0:
                    #self.reward -= 10
                enemy.enemyMovement(self.player)
        #self.gameOver()



    def playerAction(self,action):
        '''This function determines the actions of the player/agent.'''
        #the folling code determines the movement of the payer in the map.
        self.reward = 0
        if action <= 3:
            if action == 0:
                movementType = GameEnum.MovementType.up
            if action == 1:
                movementType = GameEnum.MovementType.right
            if action == 2:
                movementType = GameEnum.MovementType.down
            if action == 3:
                movementType = GameEnum.MovementType.left
            Xposition = self.player.currentPositionX
            Yposition = self.player.currentPositionY
            inventory = len(self.player.inventory)
            unknownTille = self.map.countUknownTile()
            self.player.playerMovement(movementType)
            if Xposition != self.player.currentPositionX or Yposition != self.player.currentPositionY:
                if unknownTille > self.map.countUknownTile():
                    self.reward += 1
                else:
                    self.reward -= 0
                if len(self.player.inventory) > inventory:
                    self.reward += 10
                self.map.createEnemies(self.player,movementType)
                self.enemyMove()
                self.enterCave(self.cave)
                self.afterMoveDepiction()
            else:
                self.reward -= 3       
        if action == 4:
        #This code chunk adds 4 points to the hp of the player and if the player is wizzard type, adds and 4 point the mp of the player.'''
            oldhitpoints = self.player.hitPoints
            if isinstance(self.player, Wizard):
                oldmanapoints = self.player.manaPoints
            if settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.visible and settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.fogged:
                self.player.rest()
                self.map.createEnemiesRest(self.player)
                self.enemyMove()
                self.afterMoveDepiction()
            if  (abs(settings.exitx - self.player.currentPositionX) + abs(settings.exity - self.player.currentPositionY)) > 35 and settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.unknown:
                self.player.rest()
                self.map.createEnemiesRest(self.player)
                self.enemyMove()
                self.afterMoveDepiction()
            if (abs(settings.exitx - self.player.currentPositionX) + abs(settings.exity - self.player.currentPositionY)) < 35 and settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.unknown:
                text =self.player.name + " don't try to cheat."
                settings.addGameText(text)
                self.afterMoveDepiction()
            if (self.player.hitPoints - oldhitpoints) == 4 and self.countHealthPotion() == 0:
                self.reward += 1
            if (self.player.hitPoints - oldhitpoints) < 4 or self.countHealthPotion() == 0:
                self.reward -= 3
            if isinstance(self.player, Wizard):
                if  (self.player.manaPoints - oldmanapoints) == 4 and self.countManaPotion() == 0:
                    self.reward += 1
                if  (self.player.manaPoints - oldmanapoints) < 4 or self.countManaPotion() == 0:
                    self.reward -= 1

        if action == 5:
        #This code chunk consumes one health potion and adds 20 point to the player hp.
            if self.player.inventory != []:
                index = None
                for i in range(len(self.player.inventory)):
                    if isinstance(self.player.inventory[i], HelthPotion):
                        index = i
                if index != None:
                    item = self.player.inventory.pop(index)
                    oldhitpoints = self.player.hitPoints
                    self.player.use(item)           
                if (self.player.hitPoints - oldhitpoints) == 20:
                    self.reward += 1
                if (self.player.hitPoints - oldhitpoints) < 20:
                    self.reward -= 1
            else :
                text = self.player.name + " doesn't posses health potion."
                settings.addGameText(text)
                self.reward -= 1
            self.enemyMove()
            self.afterMoveDepiction()

        if action == 6:
        #This code chunk consumes one mana potion and adds 20 points to the player mp, if the payer is a wizzard.
            if isinstance(self.player, Warrior):
                    text =self.player.name + " can't uses mana potion."
                    settings.addGameText(text)
                    self.reward -= 30

            if isinstance(self.player, Wizard) and self.player.inventory != []:
                index = None          
                for i in range(len(self.player.inventory)):
                    if isinstance(self.player.inventory[i],ManaPotion):
                        index = i
                if index != None:
                    item = self.player.inventory.pop(index)
                    oldmanapoints = self.player.manaPoints
                    self.player.use(item)
                if  (self.player.manaPoints - oldmanapoints) == 20:
                    self.reward += 1
                if  (self.player.manaPoints - oldmanapoints) < 20:
                    self.reward -= 1
                if index == None:
                    text =self.player.name + " doesn't posses mana potion."
                    settings.addGameText(text)
                    self.reward -= 1

            self.enemyMove()
            self.afterMoveDepiction()   
                
        if action == 7:
        #This code chunk performs the attack of the player.
            index = self.player.enemyToAttack()
            if index == None:
                self.reward -= 5
            if index != None:
                enemy = settings.enemies[index]
                boolean = self.player.attack(enemy)
                if boolean:
                    self.reward += 0
                if boolean and enemy.hitPoints <= 0:
                    self.reward += 10
                if boolean and enemy.hitPoints <= 0 and self.player.experiencePoints <= 13999:
                    settings.tiles[enemy.enemyCurrentPossitionX][enemy.enemyCurrentPossitionY].occupancy = False
                    settings.enemies.pop(index)
                    text = self.player.name + " kills " + enemy.name + " and earn " + str(enemy.XPreturn) + " XP points"
                    settings.addGameText(text)
                    old_level = self.player.getLevel()
                    self.player.addXP(enemy.XPreturn)
                    if old_level < self.player.getLevel() and self.player.hitPoints <= (self.player.maxHitPoints*(2/3)):
                        for i in range(self.countHealthPotion()//2): 
                            index = None
                            for i in range(len(self.player.inventory)):
                                if isinstance(self.player.inventory[i], HelthPotion):
                                    index = i
                                
                            if index != None:
                                item = self.player.inventory.pop(index)
                                self.player.use(item)
                if random.random() <= 0.25:
                    self.map.addItem(self.player, enemy.enemyCurrentPossitionX, enemy.enemyCurrentPossitionY)
            self.enemyMove()
            self.afterMoveDepiction()

        if action == 8:
            #This code chunk allows the player to pick weapon from the map.'''
            if settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store != None:
                weaponPicked = False
                if isinstance(self.player, Warrior) and isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Sword):
                    sword = settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store
                    settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store = self.player.dropWeapon()
                    weaponPicked = True
                    self.player.setWeapon(sword)
                    self.reward += 1
                if isinstance(self.player, Wizard) and isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Staff):
                    staff = settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store
                    settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store = self.player.dropWeapon()
                    weaponPicked = True
                    self.player.setWeapon(staff)
                    self.reward += 1
                if not weaponPicked:
                    self.reward -= 1
            else:
                self.reward -= 1 
            self.enemyMove()
            self.afterMoveDepiction()
        #if self.player.hitPoints <= 0:
            #self.reward -= 15
        screen = self.screen
        state = pygame.surfarray.array3d(screen)
        state = state.swapaxes(0,1)
        state = state[:settings.mapHeigth,:settings.mapWidth]
        state = state/255
        state =  np.expand_dims(state, axis=0)
        std_reward = self.standardize_reward(self.reward)
        #print(state.shape)
        return state, self.reward

    def enterCave(self,cave):
        '''This function Checks if the tile is stair or the tile stores the Werdna Ring. 
        If the tile is stair the player enters a new cave if the tile posses the Werdna ring the player wins and the game ends.'''                
        if settings.tiles[self.player.currentPositionX][self.player.currentPositionY].ground == GameEnum.GroundType.stairs:
            self.cave = self.cave + 1
            self.makeMap(self.cave)
            settings.enemies = []
            text =self.player.name + " enters cave No " + str(cave + 1)
            settings.addGameText(text)
            self.reward += 100
        if isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Werdna_Ring):
            print(self.player.name + " found Werdna's Ring!! Congratulation")
            pygame.quit()
            self.reward += 1000
            #sys.exit()
                    
    def settingmapWidth(self):
        '''This function returns the pixels of the map x axis.'''
        return settings.mapWidth


    def settingmapHeigth(self):
        '''This function returns the pixels of the map y axis.'''
        return settings.mapHeigth

    def standardize_reward(self,reward):
        self.total_reward.append(reward)
        #mean_reward = self.total_reward/episode
        #print(mean_reward)
        #print(episode)
        reward_std = ((reward - np.mean(self.total_reward)) / (np.std(self.total_reward) + eps))
        return reward_std



