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
import numpy as np
#import tensorflow as tf
eps = np.finfo(np.float32).eps.item()

class GamePAI():
    '''GamePAi is a class that creates an instance of the game and initialize the basic features of the game.'''
    def __init__(self,playerType,playerName,xPixel,yPixel,screenFactor,depict,game,playHP,seeded):
        self.seed = game
        self.additemRandom = game
        self.seeded = seeded
        self.playHP = playHP
        self.xPixel = xPixel
        self.yPixel = yPixel
        self.screenFactor = screenFactor
        self.cave = 0
        self.playerType = playerType
        self.playerName = playerName
        self.total_reward = []
        self.buffer_size = 15
        self.killNo = 0
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
        self.rest = 0
        self.reward = 0
        #random.seed(seed)
        #if self.screenFactor == 3:
        if depict == True:
            self.screen = pygame.display.set_mode((settings.mapWidth+150, settings.mapHeigth+70))
        if depict == False:
            self.screen = pygame.Surface((settings.mapWidth+150, settings.mapHeigth+70))
        #if self.screenFactor == 1:
            #if depict == True:
                #self.screen = pygame.display.set_mode((settings.mapWidth+32, settings.mapHeigth+24))
            #if depict == False:
                #self.screen = pygame.Surface((settings.mapWidth+32, settings.mapHeigth+24))
        self.DrawMap = GameMapGraphics(self.screen)
        self.map = GameMap(self.seeded,self.seed)
        pygame.init()         
        if not self.playHP:
            settings.enemies = []
            settings.game_text = []
            if playerType == 1:
                self.player = Warrior()
            if playerType == 2:
                self.player = Wizard()
            self.player.name = playerName
        if self.playHP:
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render("Please select the type of the player (Warrior or Wizard). Press 1 for Warrior or 2 for Wizard.", True, (255,0,0), (0,0,0))
            textRect = text.get_rect()
            textRect.center = ((settings.mapWidth+150)/2,(settings.mapHeigth+70)/2)
            self.screen.blit(text, textRect)
            pygame.display.set_caption("Wizard Werdna Ring")
            pygame.display.flip()
            writeText = True
            while writeText:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
            
                        if event.key == pygame.K_1:
                            self.player = Warrior()
                            writeText = False
                        if event.key == pygame.K_2:
                            self.player = Wizard()
                            writeText = False   

            text = font.render("Please enter the name of the player and press enter: ", True, (255,0,0), (0,0,0))
            textRect = text.get_rect()
            input_rect = pygame.Rect((settings.mapWidth+150)/2-100, (settings.mapHeigth+70)/2 + 20, 200, 20)
            textRect.center = ((settings.mapWidth+150)/2,(settings.mapHeigth+70)/2)
            self.screen.fill(pygame.Color("black"))
            user_text = ''
            self.screen.blit(text, textRect)
            pygame.display.set_caption("Wizard Werdna Ring")
            pygame.display.flip()
            writeText = True
            while writeText:
                nameinput = True
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            while nameinput:
                                self.player.name = user_text
                                nameinput = False
                                writeText = False           

                        if event.key == pygame.K_BACKSPACE:
                            user_text = user_text[:-1]
  
                        if event.key != pygame.K_BACKSPACE and event.key != pygame.K_RETURN:
                            user_text += event.unicode
                pygame.draw.rect(self.screen, (255,255,0), input_rect)
                text_surface = font.render(user_text, True, (255, 0, 0))
                self.screen.blit(text_surface, (input_rect.x+5, input_rect.y+5))
                input_rect.w = max(100, text_surface.get_width()+10)
                pygame.display.flip()
        text = self.player.name + ' welcome to the Wizard Werdna Ring adventure. Try to find the ring and win the Game.'
        settings.addGameText(text)
        self.printText()
        self.makeMap(self.cave)
        self.printPlayerStatus()
        if depict == True:
            if self.screenFactor == 1:
                pygame.display.set_caption(str(game))
                pygame.display.flip()
            else:
                pygame.display.set_caption("Wizard Werdna Ring "+str(game+1))
                pygame.display.flip()
        if playHP:
            self.run_gameHP()

    def afterMoveDepiction(self):
        '''This function refreshes the information on the game screen'''
        self.printText()
        self.printPlayerStatus()
        self.drawMap() 

    def gameOverHP(self):
        '''Checks if the player is dead and reinitilize the game creating a new instance of the GamePAI class.'''
        if self.player.hitPoints <= 0:
            print(self.player.name,'is dead. Game Over ' + str(settings.gameCount) + ' episode rewards ' + str(self.reward))
            pygame.quit()
            sys.exit()
            

    def gameOver(self,s):
        '''Checks if the player is dead and reinitilize the game creating a new instance of the GamePAI class.'''
        done = False
        if self.player.hitPoints <= 0:
            done = True
            #self.reward -= 10
            settings.gameCount = settings.gameCount + 1
            #print(self.player.name,'is dead. Game Over ' + str(settings.gameCount) + ' episode rewards ' + str(self.reward))
            pygame.quit()
            #sys.exit()
            if s % 100 == 0:
                self.__init__(self.playerType,self.playerName,self.xPixel,self.yPixel,self.screenFactor,True,s,False)
            else:
                self.__init__(self.playerType,self.playerName,self.xPixel,self.yPixel,self.screenFactor,True,s,False)
            gc.collect()
        return done

    def drawMap(self):
        '''This function draws the map of the game'''
        self.DrawMap.drawMap()
        self.DrawMap.drawItem()
        self.DrawMap.enemyDepiction()
        self.DrawMap.drawPlayer(self.player.currentPositionX, self.player.currentPositionY)
        FramePerSec = pygame.time.Clock()
        FramePerSec.tick(5)
        if self.depict == True:
            pygame.display.update()

    def makeMap(self,cave):
        '''Creates the map of the game.'''
        precentage = 0
        makemap = True
        if self.seeded:
            seed = self.seed + cave*10000
            while makemap:
                random.seed(seed)
                precentage = random.random()
                # print(seed,precentage)
                if 0.35 < precentage <= 0.55:
                    makemap = False
                else:
                    seed += 1
        else:
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
        text9 = font.render('Cave: ' + str(self.cave), True, (255,0,0), (0,0,0))
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
        self.screen.blit(text9, (settings.mapWidth,138))



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
                if enemy.minDistance(self.player,enemy.enemyCurrentPossitionX,enemy.enemyCurrentPossitionY) == 0:
                    self.reward -= 1
                enemy.enemyMovement(self.player)
        if self.playHP:
            self.gameOverHP()



    def playerAction(self,action):
        '''This function determines the actions of the player/agent.'''
        #the folling code determines the movement of the payer in the map.
        self.reward = 0
        initialHitPoint = self.player.hitPoints
        if isinstance(self.player, Wizard):
            initialManaPoints = self.player.manaPoints
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
                self.steps += 1
                # print(self.steps)
                if unknownTille > self.map.countUknownTile():
                    self.reward += 20
                else:
                    self.reward -= 0
                if len(self.player.inventory) > inventory:
                    self.reward += 100
                self.map.createEnemies(self.player,movementType,self.steps)
                self.enemyMove()
                self.enterCave(self.cave)
                self.afterMoveDepiction()
                if settings.tiles[settings.exitx][settings.exity].visibility == GameEnum.VisibilityType.visible:
                    self.reward += 0
            if Xposition == self.player.currentPositionX and Yposition == self.player.currentPositionY:
                self.reward -= 5
                self.enemyMove()  
                self.afterMoveDepiction()   
        if action == 4:
        #This code chunk adds 4 points to the hp of the player and if the player is wizzard type, adds and 4 point the mp of the player.'''
            oldhitpoints = self.player.hitPoints
            self.rest += 1
            # print(self.rest)
            if isinstance(self.player, Wizard):
                oldmanapoints = self.player.manaPoints
            if settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.visible and settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.fogged:
                self.player.rest()
                self.map.createEnemiesRest(self.player,self.rest)
                self.enemyMove()
                self.afterMoveDepiction()
            if  (abs(settings.exitx - self.player.currentPositionX) + abs(settings.exity - self.player.currentPositionY)) > 35 and settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.unknown:
                self.player.rest()
                self.map.createEnemiesRest(self.player,self.rest)
                self.enemyMove()
                self.afterMoveDepiction()
            if (abs(settings.exitx - self.player.currentPositionX) + abs(settings.exity - self.player.currentPositionY)) < 35 and settings.tiles[settings.exitx][settings.exity].visibility != GameEnum.VisibilityType.unknown:
                text =self.player.name + " don't try to cheat."
                settings.addGameText(text)
                self.afterMoveDepiction()
            if (self.player.hitPoints - oldhitpoints) == 4 and self.countHealthPotion() == 0:
                self.reward += 10
            if (self.player.hitPoints - oldhitpoints) < 4 or self.countHealthPotion() == 0:
                self.reward -= 0.1
            if isinstance(self.player, Wizard):
                if  (self.player.manaPoints - oldmanapoints) == 4 and self.countManaPotion() == 0:
                    self.reward += 10
                if  (self.player.manaPoints - oldmanapoints) < 4 or self.countManaPotion() == 0:
                    self.reward -= 0.1

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
                    self.reward += 10
                if (self.player.hitPoints - oldhitpoints) < 20:
                    self.reward -= 0.1
            else :
                text = self.player.name + " doesn't posses health potion."
                settings.addGameText(text)
                self.reward -= 0.1
            self.enemyMove()
            self.afterMoveDepiction()

        if action == 6:
        #This code chunk consumes one mana potion and adds 20 points to the player mp, if the payer is a wizzard.
            if isinstance(self.player, Warrior):
                    text =self.player.name + " can't uses mana potion."
                    settings.addGameText(text)
                    self.reward -= 0.1

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
                    self.reward += 10
                if  (self.player.manaPoints - oldmanapoints) < 20:
                    self.reward -= 0.1
                if index == None:
                    text =self.player.name + " doesn't posses mana potion."
                    settings.addGameText(text)
                    self.reward -= 0.1

            self.enemyMove()
            self.afterMoveDepiction()   
                
        if action == 7:
        #This code chunk performs the attack of the player.
            index = self.player.enemyToAttack()
            if index == None:
                self.reward -= 1
            if index != None:
                enemy = settings.enemies[index]
                boolean = self.player.attack(enemy)
                if boolean:
                    self.reward += 10
                if boolean and enemy.hitPoints <= 0:
                    self.killNo += 1
                    self.additemRandom += 100
                    # print(self.additemRandom)
                    self.reward += 100
                    if self.seeded:
                        random.seed(self.additemRandom)
                    r = random.random()
                    # print(r)
                    if r <= 0.25:
                        self.map.addItem(self.player, enemy.enemyCurrentPossitionX, enemy.enemyCurrentPossitionY,self.cave,self.killNo)
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
                    self.reward += 100
                if isinstance(self.player, Wizard) and isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Staff):
                    staff = settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store
                    settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store = self.player.dropWeapon()
                    weaponPicked = True
                    self.player.setWeapon(staff)
                    self.reward += 100
                if not weaponPicked:
                    self.reward -= 0.1
            else:
                self.reward -= 0.1
            self.enemyMove()
            self.afterMoveDepiction()
        if self.player.hitPoints <= 0:
            self.reward -= 1
        screen = self.screen
        state = pygame.surfarray.array3d(screen)
        state = state.swapaxes(0,1)
        state = state[:settings.mapHeigth,:settings.mapWidth]
        state = state/255
        #state =  np.expand_dims(state, axis=0)
        #std_reward = self.standardize_reward(self.reward)
        manapoints = 0
        maxmanapoints = 0
        manapotion = 0
        inteligence = 0
        if isinstance(self.player,Warrior):
            base_int_str = self.player.baseStrength
        if isinstance(self.player, Wizard):
            manapoints = self.player.manaPoints
            maxmanapoints = self.player.maxManaPoints
            manapotion = self.countManaPotion()
            base_int_str = self.player.baseIntelligence
        playerstatus = np.array([self.player.hitPoints,self.player.maxHitPoints,base_int_str,inteligence,manapoints,maxmanapoints,manapotion,self.countHealthPotion(),self.player.getAttackDamage(),self.player.getLevel()],dtype=np.int32)
        playerstatus = playerstatus/(max(playerstatus))
        textList = []
        for text in settings.game_text:
            textNname = text[len(self.player.name):]
            textList.append(textNname)        
        #playerstatus = np.expand_dims(playerstatus, axis=0)
        textArray = self.gameVocab(textList)
        #reward = self.standardize_reward(self.reward)
        # if len(self.buffer_playerStatus) > self.buffer_size:
            # del self.buffer_playerStatus[0]
        # if len(self.buffer_playerStatus)<=self.buffer_size:
            # for i in range(3-len(self.buffer_playerStatus)):
                # self.buffer_playerStatus.append(playerstatus)
        # self.buffer_playerStatus.append(playerstatus)
        # array_playerStatus = np.array(self.buffer_playerStatus)
        # print(array_playerStatus.shape)
        # array_playerStatus = np.expand_dims(array_playerStatus, axis=0)

        # if len(self.buffer_State) > self.buffer_size:
            # del self.buffer_State[0]
        # if len(self.buffer_State)<=self.buffer_size:
            # for i in range(self.buffer_size-len(self.buffer_State)):
                # self.buffer_State.append(state)
        # self.buffer_State.append(state)
        # array_state = np.array(self.buffer_State)
        # array_state = np.expand_dims(array_state, axis=0)

        # if len(self.buffer_text) > self.buffer_size:
            # del self.buffer_text[0]
        # if len(self.buffer_text)<=self.buffer_size:
            # for i in range(3-len(self.buffer_text)):
                # self.buffer_text.append(textArray)
        # self.buffer_text.append(textArray)
        # array_text = np.array(self.buffer_text)
        # array_text = np.expand_dims(array_text, axis=0)
        done = False
        if self.player.hitPoints <= 0:
            done = True
        if initialHitPoint < self.player.hitPoints:
            self.reward += 1
        if isinstance(self.player,Wizard):
            if initialManaPoints < self.player.manaPoints:
                self.reward += 1
        # print(np.array(self.buffer_text).shape)
        # if len(self.buffer_reward) > self.buffer_size:
            # del self.buffer_reward[0]
        # print(self.reward)
        # self.buffer_reward.append(self.reward)
        # print(self.buffer_reward[len(self.buffer_reward)-1])
        # print(self.buffer_reward)
        return state, playerstatus,  textArray, self.reward, done

    def gameVocab(self,textList):
        array = np.array([])
        list1 = []
        list2 = []
        #sum_text = 0
        game_vocab = {'welcome':100,'to':1,'the':2,'Wizard':3,'Werdna':4,'Ring':5,'adventure.':6,
        'Try':7,'to':8,'find':9,'Ring':10,'and':11,'win':12,'Game.':13,"don't":14,'try':15,'cheat.':16,
        "doesn't":17,'posses':18,'health':19,'potion.':20,"can't":21,'uses':22,'mana':22,'kills':23,'earn':24,
        'XP':25,'points':26,'enters':27,'cave':28,'No':29,'can':30,'hear':31,'from':32,'east':33,
        'west':34,'north':35,'south':36,'attacked':37,'by':38,'for':39,'damage.':40,'a':41,'an':42,
        'Giant':43,'Rat':44,'Goblin':45,'Slime':46,'Orc':47,'Grunt':48,'Warlord':49,'Skeleton':50,
        'Vampire':51,'Wyrm':52,'changes':53,'level.':54,'New':55,'level':56,'is':57,'HP.':58,'potion':59,
        'MP.':60,'found':61,'press':63,'use':64,'it.':65,'h':66,'m':67,'weapon':68,'type':69,'of':70,
        'p':71,'equip':72,'with':73,'add':74,'which':75,'add':76,'max':77,'HP':78,'MP':79,'player':80,'inteligence':81,
        'earn':82,'attack':83,'strength':84,'east.':85,'west.':86,'north.':87,'south.':89,'ring':90,'Mana':91,'this':92,'Gray':93,
        'Amazing':94,'Deadly':95,'Ancient':96,'Sword':97,'Staff':98,'weapon.':99,'rest':101}
        # this code chunk creates a sum input for the model.
        # text = textList[len(textList)-1]
        # text = text.split()
        # for word in text:
            # if word in game_vocab:
               # sum_text += game_vocab[word]
            # else:
                # sum_text += int(word)
        # array_text = np.array(sum_text,dtype=np.int32)
        # array_text = np.reshape(array_text,(1,1))
        #print(array_text,array_text.shape)
        # return array_text
        for text in textList:
            text = text.split()
            list1 = []
            for word in text:
                if word in game_vocab:
                    list1.append(game_vocab[word])
                else:
                    list1.append(int(word))
            if len(list1) < 25:
                for i in range((25-len(list1))):
                    list1.insert(len(list1)+1,0)
            list2.append(list1)
        if len(list2)<5:
            for i in range (5-len(list2)):
                list2.append([0]*25)
        array = np.array(list2,dtype=np.int32)
        #array = np.reshape(array,(1,125))
        #array = np.expand_dims(array, axis=0)
        #array = np.expand_dims(array, axis=0)
        #print(type(array))
        #tfarray = tf.convert_to_tensor(array)
        array = np.reshape(array,(125))
        array = array/101
        #print(sumarray.shape,sumarray)
        return array

    def enterCave(self,cave):
        '''This function Checks if the tile is stair or the tile stores the Werdna Ring. 
        If the tile is stair the player enters a new cave if the tile posses the Werdna ring the player wins and the game ends.'''                
        if settings.tiles[self.player.currentPositionX][self.player.currentPositionY].ground == GameEnum.GroundType.stairs:
            self.cave = self.cave + 1
            self.makeMap(self.cave)
            settings.enemies = []
            text =self.player.name + " enters cave No " + str(cave + 1)
            settings.addGameText(text)
            self.reward += 10000
        if isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Werdna_Ring):
            print(self.player.name + " found Werdna's Ring!! Congratulation")
            pygame.quit()
            self.reward += 10000
            #sys.exit()
                    
    def settingmapWidth(self):
        '''This function returns the pixels of the map x axis.'''
        return settings.mapWidth


    def settingmapHeigth(self):
        '''This function returns the pixels of the map y axis.'''
        return settings.mapHeigth

    def standardize_reward(self,reward):
        '''This function is used for the standardization of the reward.'''
        self.total_reward.append(reward)
        #mean_reward = self.total_reward/episode
        #print(mean_reward)
        #print(episode)
        reward_std = ((reward - np.mean(self.total_reward)) / (np.std(self.total_reward) + eps))
        return reward_std

    def run_gameHP(self):
        '''Thru this function the player can play the game with the keyboard'''
        self.drawMap()
        gameContinues = True
        while gameContinues:
            settings.reward = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.playerAction(3)
                    if event.key == pygame.K_s:
                        self.playerAction(2)
                    if event.key == pygame.K_d:
                        self.playerAction(1)
                    if event.key == pygame.K_w:
                        self.playerAction(0)
                    if event.key == pygame.K_r:
                        self.playerAction(4)
                    if event.key == pygame.K_h:
                        self.playerAction(5)
                    if event.key == pygame.K_m:
                        self.playerAction(6)
                    if event.key == pygame.K_SPACE:
                        self.playerAction(7)
                    if event.key == pygame.K_p:
                        self.playerAction(8)
                    

    
    def initialGameState(self):
        '''Returns the intial state of the game'''
        #print(len(self.buffer_State),len(self.buffer_text),len(self.buffer_playerStatus))
        self.drawMap()
        screen = self.screen
        state = pygame.surfarray.array3d(screen)
        state = state.swapaxes(0,1)
        state = state[:settings.mapHeigth,:settings.mapWidth]
        state = state/255
        #state =  np.expand_dims(state, axis=0)
        #std_reward = self.standardize_reward(self.reward)
        manapoints = 0
        maxmanapoints = 0
        manapotion = 0
        inteligence = 0
        if isinstance(self.player,Warrior):
            base_int_str = self.player.baseStrength
        if isinstance(self.player, Wizard):
            manapoints = self.player.manaPoints
            maxmanapoints = self.player.maxManaPoints
            manapotion = self.countManaPotion
            base_int_str = self.player.baseIntelligence
        playerstatus = np.array([self.player.hitPoints,self.player.maxHitPoints,base_int_str,inteligence,manapoints,maxmanapoints,manapotion,self.countHealthPotion(),self.player.getAttackDamage(),self.player.getLevel()])
        playerstatus = playerstatus/max(playerstatus)
        textList = []
        for text in settings.game_text:
            textNname = text[len(self.player.name):]
            textList.append(textNname)
        textArray = self.gameVocab(textList)

        # if len(self.buffer_playerStatus)<=self.buffer_size:
            # for i in range(self.buffer_size-len(self.buffer_playerStatus)):
                # self.buffer_playerStatus.append(playerstatus)
        # self.buffer_playerStatus.append(playerstatus)
        # array_playerStatus = np.array(self.buffer_playerStatus)
        # array_playerStatus = np.expand_dims(array_playerStatus, axis=0)
        # if len(self.buffer_State)<=self.buffer_size:
            # for i in range(self.buffer_size-len(self.buffer_State)):
                # self.buffer_State.append(state)
        # self.buffer_State.append(state)
        # array_state = np.array(self.buffer_State)
        # array_state = np.expand_dims(array_state, axis=0)

        # if len(self.buffer_text)<=self.buffer_size:
            # for i in range(self.buffer_size-len(self.buffer_text)):
                # self.buffer_text.append(textArray)
        # self.buffer_text.append(textArray)
        # array_text = np.array(self.buffer_text)
        # array_text = np.expand_dims(array_text, axis=0)

        
        #textArray = np.expand_dims(textArray, axis=0)
        return state, playerstatus, textArray



if __name__ == '__main__':
    game = GamePAI(1,'Connan',444,444,3,True,0,True,True)
    
