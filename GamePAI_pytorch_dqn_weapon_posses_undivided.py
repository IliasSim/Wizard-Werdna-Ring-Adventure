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
import os
import matplotlib.pyplot as plt

#import tensorflow as tf
eps = np.finfo(np.float32).eps.item()

class GamePAI():
    '''GamePAi is a class that creates an instance of the game and initialize the basic features of the game.'''
    def __init__(self,playerType,playerName,xPixel,yPixel,screenFactor,depict,game,playHP,seeded,agent_i):
        self.agent_i = agent_i
        if agent_i<=4:
            x= 300*agent_i + 200
            y= 100
        if agent_i>=4:
            x= 300*agent_i - 1000
            y= 400
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d"%(x,y)
        self.game = game
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
        self.reward_useless_action = 0
        self.reward_usefull_action = 1
        self.consecutive_hp = 0
        self.consecutive_attack = 0
        self.consecutive_pick = 0
        self.consecutive_rest = 0
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
        if self.depict == True:
            self.screen = pygame.display.set_mode((settings.mapWidth+150, settings.mapHeigth+70))
        if self.depict == False:
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
            if self.playerType == 1:
                self.player = Warrior()
            if self.playerType == 2:
                self.player = Wizard()
            self.player.name = self.playerName
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
        if self.depict == True:
            if self.screenFactor == 1:
                pygame.display.set_caption(str(self.game) + ' agent ' + str(self.agent_i))
                pygame.display.flip()
            else:
                pygame.display.set_caption("Wizard Werdna Ring "+str(self.game+1))
                pygame.display.flip()
        if self.playHP:
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
            settings.gameCount = settings.gameCount + 1
            pygame.quit()
            if s % 100 == 0:
                self.__init__(self.playerType,self.playerName,self.xPixel,self.yPixel,self.screenFactor,True,s,False)
            else:
                self.__init__(self.playerType,self.playerName,self.xPixel,self.yPixel,self.screenFactor,True,s,False)
            gc.collect()
        return done

    def gameOver_pytorch(self):
        '''Quit game for pytorch script'''
        pygame.quit()
        gc.collect()

    def drawMap(self):
        '''This function draws the map of the game'''
        self.DrawMap.drawMap()
        self.DrawMap.drawItem()
        self.DrawMap.enemyDepiction()
        self.DrawMap.drawPlayer(self.player.currentPositionX, self.player.currentPositionY)
        # FramePerSec = pygame.time.Clock()
        # FramePerSec.tick(15)
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
                enemy.enemyMovement(self.player)
        if self.playHP:
            self.gameOverHP()



    def playerAction(self,action):
        '''This function determines the actions of the player/agent.'''
        #the folling code determines the movement of the payer in the map.
        self.reward = 0
        bool_player_move = False
        attack_boolean = False
        kill_boolean = False
        unknownTille = self.map.countUknownTile()
        weaponPicked = False
        initialHitPoint = self.player.hitPoints
        oldhitpoints = self.player.hitPoints
        oldmanapoints = 0
        inventory = len(self.player.inventory)
        enter_cave_boolean = False
        if isinstance(self.player, Wizard):
            oldmanapoints = self.player.manaPoints
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
            self.player.playerMovement(movementType)
            if Xposition != self.player.currentPositionX or Yposition != self.player.currentPositionY:
                self.steps += 1
                # print(self.steps)
                bool_player_move = True
                self.map.createEnemies(self.player,movementType,self.steps)
                self.enemyMove()
                enter_cave_boolean =self.enterCave(self.cave)
                if enter_cave_boolean:
                    done = True
                self.afterMoveDepiction()      
            if Xposition == self.player.currentPositionX and Yposition == self.player.currentPositionY:
                bool_player_move = False
                
                self.enemyMove()  
                self.afterMoveDepiction()   
        if action == 4:
        #This code chunk adds 4 points to the hp of the player and if the player is wizzard type, adds and 4 point the mp of the player.'''
            self.rest += 1
            # print(self.rest)
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
            

        if action == 5:
        #This code chunk consumes one health potion and adds 20 point to the player hp.
            if self.player.inventory != []:
                index = None
                for i in range(len(self.player.inventory)):
                    if isinstance(self.player.inventory[i], HelthPotion):
                        index = i
                if index != None:
                    item = self.player.inventory.pop(index)
                    self.player.use(item)           
            else :
                text = self.player.name + " doesn't posses health potion."
                settings.addGameText(text)
            self.enemyMove()
            self.afterMoveDepiction()   
                
        if action == 6:
        #This code chunk performs the attack of the player.
            index = self.player.enemyToAttack()
            if index != None:
                enemy = settings.enemies[index]
                attack_boolean = self.player.attack(enemy)
                
                if attack_boolean and enemy.hitPoints <= 0:
                    kill_boolean = True
                    self.killNo += 1
                    self.additemRandom += 1
                    # print(self.additemRandom)
                    
                    if self.seeded:
                        random.seed(self.additemRandom)
                    r = random.random()
                    # print(r)
                    if r <= 0.25:
                        self.map.addItem(self.player, enemy.enemyCurrentPossitionX, enemy.enemyCurrentPossitionY,self.cave,self.killNo)
                if attack_boolean and enemy.hitPoints <= 0 and self.player.experiencePoints <= 13999:
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

        if action == 7:
            #This code chunk allows the player to pick weapon from the map.'''
            if settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store != None:       
                if isinstance(self.player, Warrior) and isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Sword):
                    sword = settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store
                    settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store = self.player.dropWeapon()
                    weaponPicked = True
                    self.player.setWeapon(sword)
                if isinstance(self.player, Wizard) and isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Staff):
                    staff = settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store
                    settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store = self.player.dropWeapon()
                    weaponPicked = True
                    self.player.setWeapon(staff)
            self.enemyMove()
            self.afterMoveDepiction()

        if action == 8:
        #This code chunk consumes one mana potion and adds 20 points to the player mp, if the payer is a wizzard.
            if isinstance(self.player, Warrior):
                    text =self.player.name + " can't uses mana potion."
                    settings.addGameText(text)

            if isinstance(self.player, Wizard) and self.player.inventory != []:
                index = None          
                for i in range(len(self.player.inventory)):
                    if isinstance(self.player.inventory[i],ManaPotion):
                        index = i
                if index != None:
                    item = self.player.inventory.pop(index)
                    oldmanapoints = self.player.manaPoints
                    self.player.use(item)
                
                if index == None:
                    text =self.player.name + " doesn't posses mana potion."
                    settings.addGameText(text)

            self.enemyMove()
            self.afterMoveDepiction()

        screen = self.screen
        state = pygame.surfarray.array3d(screen)
        state = state.swapaxes(0,1)
        state = state[:settings.mapHeigth,:settings.mapWidth]
        state = state #/255
        state = np.reshape(state,(3,settings.mapHeigth,settings.mapWidth))
        
        # state = np.zeros((3,settings.mapHeigth,settings.mapWidth))
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
        if self.player.weapon != None:
            weaponencode = 1
        else :
            weaponencode = 0
        playerstatus = np.array([weaponencode,self.player.hitPoints,self.player.maxHitPoints,self.player.getAttackDamage(),self.countHealthPotion(),self.player.getLevel(),self.player.experiencePoints,self.cave],dtype=np.int32)
        # playerstatus = np.array([weaponencode,self.player.hitPoints,self.player.maxHitPoints,self.player.maxHitPoints,100,base_int_str,self.player.getAttackDamage(),inteligence,manapoints,maxmanapoints,manapotion,self.countHealthPotion(),30,self.player.getAttackDamage(),(45+self.player.getStrengthFromEquimpment()),self.player.getLevel(),5,self.player.experiencePoints,self.maxlevelxp(self.player.getLevel())],dtype=np.float32)
        # playerstatus = playerstatus/30 # wizzard will be 20
        textList = []
        for text in settings.game_text:
            textNname = text[len(self.player.name):]
            textList.append(textNname)        
        textArray = self.gameVocab(textList)
        done = False
        if self.player.hitPoints <= 0:
            done = True
        self.calculate_reward(action,bool_player_move,attack_boolean,kill_boolean,unknownTille,weaponPicked,oldhitpoints,oldmanapoints,enter_cave_boolean,inventory)
        state = state.astype('int16')
        playerstatus = playerstatus.astype('int16')
        textArray = textArray.astype('int16')
        return state, playerstatus, textArray, self.reward, done

    def gameVocab(self,textList):
        array = np.array([])
        list1 = []
        list2 = []
        #sum_text = 0
        game_vocab = {'welcome':0,'to':1,'the':2,'Wizard':3,'Werdna':4,'Ring':5,'adventure.':6,
        'Try':7,'find':8,'and':9,'win':10,'Game.':11,"don't":12,'try':13,'cheat.':14,
        "doesn't":15,'posses':16,'health':17,'potion.':18,"can't":19,'uses':20,'mana':21,'kills':22,'earn':23,
        'XP':24,'points':25,'enters':26,'cave':27,'No':28,'can':29,'hear':30,'from':31,'east':32,
        'west':33,'north':34,'south':35,'attacked':36,'by':37,'for':38,'damage.':39,'a':40,'an':41,
        'Giant':42,'Rat':43,'Goblin':44,'Slime':45,'Orc':46,'Grunt':47,'Warlord':48,'Skeleton':49,
        'Vampire':50,'Wyrm':51,'changes':52,'level.':53,'New':54,'level':55,'is':56,'HP.':57,'potion':58,
        'MP.':59,'found':60,'press':61,'use':62,'it.':63,'h':64,'m':65,'weapon':66,'type':67,'of':68,
        'p':69,'equip':70,'with':71,'add':72,'which':73,'max':74,'HP':75,'MP':76,'player':77,'inteligence':78,
        'earn':79,'attack':80,'strength':81,'east.':82,'west.':83,'north.':84,'south.':85,'ring':86,'Mana':87,'this':88,'Gray':89,
        'Amazing':90,'Deadly':91,'Ancient':92,'Sword':93,'Staff':94,'weapon.':95,'rest':96,'Ettin':97}
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
        # array = array 
        #print(sumarray.shape,sumarray)
        return array

    def enterCave(self,cave):
        '''This function Checks if the tile is stair or the tile stores the Werdna Ring. 
        If the tile is stair the player enters a new cave if the tile posses the Werdna ring the player wins and the game ends.'''  
        enter_cave_boolean = False              
        if settings.tiles[self.player.currentPositionX][self.player.currentPositionY].ground == GameEnum.GroundType.stairs:
            self.exit_visibility = False
            self.cave = self.cave + 1
            self.makeMap(self.cave)
            settings.enemies = []
            text =self.player.name + " enters cave No " + str(cave + 1)
            settings.addGameText(text)
            enter_cave_boolean = True
        if isinstance(settings.tiles[self.player.currentPositionX][self.player.currentPositionY].store, Werdna_Ring):
            print(self.player.name + " found Werdna's Ring!! Congratulation")
            enter_cave_boolean = True
            #sys.exit()
        return enter_cave_boolean
                    
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
                        self.playerAction(8)
                    if event.key == pygame.K_SPACE:
                        self.playerAction(6)
                    if event.key == pygame.K_p:
                        self.playerAction(7)
                    

    
    def initialGameState(self):
        '''Returns the intial state of the game'''
        #print(len(self.buffer_State),len(self.buffer_text),len(self.buffer_playerStatus))
        self.drawMap()
        screen = self.screen
        state = pygame.surfarray.array3d(screen)
        state = state.swapaxes(0,1)
        state = state[:settings.mapHeigth,:settings.mapWidth]
        state = state #/255
        state = np.reshape(state,(3,settings.mapHeigth,settings.mapWidth))
        # state = np.zeros((3,settings.mapHeigth,settings.mapWidth)) #zero test
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
        if self.player.weapon != None:
            weaponencode = 1
        else :
            weaponencode = 0
        playerstatus = np.array([weaponencode,self.player.hitPoints,self.player.maxHitPoints,self.player.getAttackDamage(),self.countHealthPotion(),self.player.getLevel(),self.player.experiencePoints,self.cave],dtype=np.int16)
        textList = []
        for text in settings.game_text:
            textNname = text[len(self.player.name):]
            textList.append(textNname)
        textArray = self.gameVocab(textList)
        state = state.astype('int16')
        playerstatus = playerstatus.astype('int16')
        textArray = textArray.astype('int16')
        return state, playerstatus, textArray

    def calculate_reward(self,action,bool_player_move,attack_boolean,kill_boolean,unknownTille,weaponPicked,oldhitpoints,oldmanapoints,enter_cave_boolean,inventory):
        reward = 0
        if action <= 3:
            if bool_player_move:
                if unknownTille > self.map.countUknownTile():
                    reward = self.reward_usefull_action*20
                if unknownTille == self.map.countUknownTile():
                    reward = self.reward_usefull_action
                if settings.tiles[settings.exitx][settings.exity].visibility == GameEnum.VisibilityType.visible:
                    reward = 100
                # else:
                    # self.reward -= 0
                if len(self.player.inventory) > inventory:
                    reward = self.reward_usefull_action*1000
                # self.ItemVisibilityRewards()
                # self.EnemyVisibilityRewards(self.player) 
            if not bool_player_move:
                reward = -4 # self.reward_useless_action

        if action == 4:
            if (self.player.hitPoints - oldhitpoints) == 4:
                reward = self.reward_usefull_action*100
            if (self.player.hitPoints - oldhitpoints) < 4:
                reward = self.reward_useless_action
            if isinstance(self.player, Wizard):
                if  (self.player.manaPoints - oldmanapoints) == 4:
                    reward = self.reward_usefull_action*100
                if  (self.player.manaPoints - oldmanapoints) < 4:
                    reward = self.reward_useless_action
        
        if action == 5:    
            if (self.player.hitPoints - oldhitpoints) == 20:
                reward = self.reward_usefull_action*1000
            # if (self.player.hitPoints - oldhitpoints) < 20:
            if (self.player.hitPoints - oldhitpoints) > 0:
                reward = self.reward_usefull_action*100

        if action == 6:
            if not attack_boolean == None:
                reward = self.reward_useless_action
            if kill_boolean:
                reward = self.reward_usefull_action*100
        
        if action == 7:
            if weaponPicked:
                reward = self.reward_usefull_action*1000
            if not weaponPicked:
                reward = self.reward_useless_action
            
        if action == 8:
            if isinstance(self.player, Warrior):
                reward -= 100
            if isinstance(self.player, Wizard):    
                if  (self.player.manaPoints - oldmanapoints) >= 20:
                        reward = self.reward_usefull_action*1000
                if  (self.player.manaPoints - oldmanapoints) < 20:
                        reward = self.reward_useless_action*100
        if oldhitpoints > self.player.hitPoints:
            reward = -10
        if oldhitpoints > self.player.hitPoints and attack_boolean:
            reward = 20
        if enter_cave_boolean:
            reward = 10000
        consecutive_reward_bool = self.countConsecutiveUselesActions(action,reward)
        if consecutive_reward_bool:
            reward = -10
        self.reward = reward
        self.ItemVisibilityRewards()
        self.EnemyVisibilityRewards(self.player)

        
            

    def ItemVisibilityRewards(self):
        for i in range(settings.xtile):
            for h in range(settings.ytile):
                if settings.tiles[i][h].visibility == GameEnum.VisibilityType.visible and settings.tiles[i][h].store != None:
                    if isinstance(self.player,Wizard):
                        if isinstance(settings.tiles[i][h].store, ManaPotion) and settings.tiles[i][h].store.seen != True:
                            settings.tiles[i][h].store.seen = True
                            self.reward +=50
                        if isinstance(settings.tiles[i][h].store, Staff) or isinstance(settings.tiles[i][h].store, Sword): 
                            if settings.tiles[i][h].store.seen != True:
                                settings.tiles[i][h].store.seen = True
                                self.reward +=100
                    if isinstance(settings.tiles[i][h].store, HelthPotion) and settings.tiles[i][h].store.seen != True:
                        settings.tiles[i][h].store.seen = True
                        self.reward +=50
                    if isinstance(self.player,Warrior):
                        if isinstance(settings.tiles[i][h].store, Sword) or isinstance(settings.tiles[i][h].store, Staff): 
                            if settings.tiles[i][h].store.seen != True:
                                settings.tiles[i][h].store.seen = True 
                                self.reward +=50

    def EnemyVisibilityRewards(self,player):
        for enemy in settings.enemies:
            distance = abs(player.currentPositionX - enemy.enemyCurrentPossitionX) + abs(player.currentPositionY - enemy.enemyCurrentPossitionY)
            if distance <= settings.radius and enemy.seen != True:
                enemy.seen = True
                self.reward += 20
    
    def countConsecutiveUselesActions(self,action,reward):
        consecutive_reward_bool   = False
        if action <= 3 or reward > 0:
            self.consecutive_pick = 0
            self.consecutive_hp = 0
            self.consecutive_attack = 0
            self.consecutive_rest = 0
        if action == 4 and reward == 0:
            self.consecutive_rest += 1
            self.consecutive_hp = 0
            self.consecutive_attack = 0
            self.consecutive_pick = 0
        if action == 5 and reward == 0:
            self.consecutive_hp += 1
            self.consecutive_attack = 0
            self.consecutive_pick = 0
            self.consecutive_rest = 0
        if action == 6 and reward == 0:
            self.consecutive_attack += 1
            self.consecutive_hp = 0
            self.consecutive_pick = 0
            self.consecutive_rest = 0
        if action == 7 and reward == 0:
            self.consecutive_pick += 1
            self.consecutive_hp = 0
            self.consecutive_attack = 0
            self.consecutive_rest = 0
        
        if self.consecutive_attack >= 8:
            consecutive_reward_bool = True
        if self.consecutive_pick >= 8:
            consecutive_reward_bool = True
        if self.consecutive_hp >= 8:
            consecutive_reward_bool = True
        if self.consecutive_rest >= 8:
            consecutive_reward_bool = True
        # print(consecutive_reward_bool,self.consecutive_attack,self.consecutive_hp,self.consecutive_rest,self.consecutive_pick)
        return consecutive_reward_bool
    

    

if __name__ == '__main__': 
    game = GamePAI(1,'Connan',444,444,3,True,0,True,False,0)
    game.playGame()
    
