import GameSettings as settings
import random
import GameEnum
import math
from Warrior import Warrior
from Games_items import HelthPotion,ManaPotion,Sword,Staff,Werdna_Ring
from Enemies import Vampire,Wyrm,Giant_Rat,Goblin,Gray_Slime,Ettin,Orc_Grunt,Orc_Warlord,Skeleton
from Enemy import Enemy

class GameMap():
    '''The GameMap class create the game map. Also it creates the enemies and the items to be placed on the map.'''
    def __init__(self,seeded,seed):
        self.seed = seed
        self.seeded = seeded
    def refreshTilesSettings(self):
        '''Reset the values of the tiles to their initial price.'''
        for y in range(settings.ytile):
            for x in range(settings.xtile):
                settings.tiles[x][y].ground = GameEnum.GroundType.wall
                settings.tiles[x][y].store = None
                settings.tiles[x][y].occupancy = False
                settings.tiles[x][y].visibility = GameEnum.VisibilityType.unknown
    
    def countUknownTile(self):
        uknownTile = 0
        for y in range(settings.ytile):
            for x in range(settings.xtile):
                if settings.tiles[x][y].visibility == GameEnum.VisibilityType.unknown and  settings.tiles[x][y].ground == GameEnum.GroundType.floor:
                    uknownTile += 1
        return uknownTile
    
    def MakeMAp(self,precentage,player,caveNo):
        '''Characterizes the tile of the map as floor or wall and sets the initial visibility to unknown, except for the region the player'''
        if self.seeded:
            random.seed(self.seed + caveNo*1000)
        settings.startX = random.randint(0, settings.xtile - 1)
        settings.startY = random.randint(0, settings.ytile -1)
        # print(self.seed + caveNo*500,settings.startX,settings.startY)
        m = settings.startX
        n = settings.startY
        player.currentPositionX = m
        player.currentPositionY = n
        settings.tiles[m][n].ground = GameEnum.GroundType.floor
        settings.tiles[m][n].visibility = GameEnum.VisibilityType.visible
        settings.tiles[m][n].occupancy = True
        while settings.countfloortile/(settings.xtile*settings.ytile) < precentage:
            r = random.random()
            if r <= 0.25 and m + 1 <= settings.xtile-1:
                m = m + 1
                if settings.tiles[m][n].ground == GameEnum.GroundType.wall:
                    settings.countfloortile = settings.countfloortile + 1
                settings.tiles[m][n].ground = GameEnum.GroundType.floor
                

            if r > 0.25 and r <= 0.5 and m - 1 >= 0:
                m = m - 1
                if settings.tiles[m][n].ground == GameEnum.GroundType.wall:
                    settings.countfloortile = settings.countfloortile + 1
                settings.tiles[m][n].ground = GameEnum.GroundType.floor
                
            if r > 0.5 and r <= 0.75 and n + 1 <= settings.ytile - 1:
                n = n + 1
                if settings.tiles[m][n].ground == GameEnum.GroundType.wall:
                    settings.countfloortile = settings.countfloortile + 1
                settings.tiles[m][n].ground = GameEnum.GroundType.floor
                
            if r <= 1 and r > 0.75 and n - 1 >= 0:
                n = n - 1
                if settings.tiles[m][n].ground == GameEnum.GroundType.wall:
                    settings.countfloortile = settings.countfloortile + 1
                settings.tiles[m][n].ground = GameEnum.GroundType.floor
                
        self.MapExit(caveNo)
        self.addItems(player,caveNo)
        settings.countfloortile = 0

    def MapExit(self,caveNo):
        '''Creates the exit at caves 1 thru 9 and adds the Werdna Ring in the last cave. 
        The exit/ring is at the maximum distance from the start of the cave.'''
        dx = settings.startX
        dy = settings.startY
        distance = 0
        for y in range(settings.ytile):
             for x in range(settings.xtile):
                 if settings.tiles[x][y].ground == GameEnum.GroundType.floor:
                    new_distance = abs(dx - x) + abs(dy - y)
                    if distance < new_distance:
                        distance = new_distance
                        settings.exitx = x
                        settings.exity = y
        if caveNo == 9:
            settings.tiles[settings.exitx][settings.exity].store = Werdna_Ring()
        else:    
            settings.tiles[settings.exitx][settings.exity].ground = GameEnum.GroundType.stairs

    def addItem(self,player,x,y,caveNo,killno):
        '''Adds item at the map when an enemy killed.'''
        if self.seeded:
            random.seed(self.seed + 3999 + caveNo + killno)
        if settings.tiles[x][y].store == None:
            r1 = random.random()
            # print(r1,1,self.seed + 3999 + caveNo + killno)
            if r1 <=0.25:
                settings.tiles[x][y].store = self.weaponGenerator(player,killno,x,y)
            else:
                store = None
                r2 = random.random()
                # print(r2,2,self.seed + 3999 + caveNo + killno)
                if r2 <= 0.5:
                    store = HelthPotion(x,y)
                else:
                    store = ManaPotion(x,y)
                settings.tiles[x][y].store = store


    def addItems(self,player,caveNo):
        '''Adds items at the creation of the map.'''
        items = 0
        store = None
        floorTile = []
        for y in range(settings.ytile):
            for x in range(settings.xtile):
                if settings.tiles[x][y].ground == GameEnum.GroundType.floor:
                    floorTile.append([x,y])
        if len(floorTile)/(settings.xtile*settings.ytile)<0.4:
            items = 5
        if len(floorTile)/(settings.xtile*settings.ytile)>0.4:
            items = 10
        for i in range(items):
            if self.seeded:
                random.seed(self.seed + caveNo*i + 4999)
            store_place = floorTile[random.randint(0,len(floorTile)-1)]
            if random.random() <= 0.5:
                store = HelthPotion(store_place[0],store_place[1])
            else:
                store = ManaPotion(store_place[0],store_place[1])
            settings.tiles[store_place[0]][store_place[1]].store = store
        if len(floorTile)/(settings.xtile*settings.ytile)<0.4:
            items = 1
        if len(floorTile)/(settings.xtile*settings.ytile)>0.4:
            items = 2
        for i in range(items):
            if self.seeded:
                random.seed(self.seed + caveNo*i + 5999)
            store_place = floorTile[random.randint(0,len(floorTile)-1)]
            # self.weaponGenerator(player,caveNo)
            settings.tiles[store_place[0]][store_place[1]].store = self.weaponGenerator(player,caveNo,store_place[0],store_place[1])
            

    def weaponGenerator(self,player,seedCH,x,y):
        '''Generates weapon to be placed at the creation of the map or after the death of an enemy
	    The item effect of the weapon depends on the player level.'''
        if self.seeded:
            random.seed(self.seed + seedCH*1000 + 5999)
        hpboostwar,hpboostwiz,manaboost,strenghtboost,intboost,totalboost = 0,0,0,0,0,0
        level = player.getLevel()
        adjective = None
        weapon = None
        if level == 1:
            totalboost = 10
            hpboostwar = random.randint(0, totalboost)
            strenghtboost = totalboost - hpboostwar
            hpboostwiz = random.randint(0, totalboost)
            manaboost = random.randint(0, totalboost - hpboostwiz)
            intboost =  totalboost - hpboostwiz - manaboost
        if level == 2:
            totalboost = 20
            hpboostwar = random.randint(0, totalboost)
            strenghtboost = totalboost - hpboostwar
            hpboostwiz = random.randint(0, totalboost)
            manaboost = random.randint(0, totalboost - hpboostwiz)
            intboost =  totalboost - hpboostwiz - manaboost
            
        if level == 3:
            totalboost = 30
            hpboostwar = random.randint(0, totalboost)
            strenghtboost = totalboost - hpboostwar
            hpboostwiz = random.randint(0, totalboost)
            manaboost = random.randint(0, totalboost - hpboostwiz)
            intboost =  totalboost - hpboostwiz - manaboost
            
        if level == 4:
            totalboost = 40
            hpboostwar = random.randint(0, totalboost)
            strenghtboost = totalboost - hpboostwar
            hpboostwiz = random.randint(0, totalboost)
            manaboost = random.randint(0, totalboost - hpboostwiz)
            intboost =  totalboost - hpboostwiz - manaboost
            
        if level == 5:
            totalboost = 60
            hpboostwar = random.randint(0, totalboost)
            strenghtboost = totalboost - hpboostwar
            hpboostwiz = random.randint(0, totalboost)
            manaboost = random.randint(0, totalboost - hpboostwiz)
            intboost =  totalboost - hpboostwiz - manaboost
            
        namep = random.random()
        if namep <= 1/3:
            adjective = "Amazing"
        if namep > 1/3 and namep <= 2/3:
            adjective = "Deadly"
        if namep > 2/3:
            adjective = "Ancient"
        if random.random() <= 0.5:
            weapon = Sword(hpboostwar, strenghtboost, adjective + " Sword",x,y)
        else:
            weapon = Staff(hpboostwiz, manaboost, intboost, adjective + " Staff",x,y)
        return weapon

    def nearestEnemy(self,player):
        '''Returns the enemy with the minimum distance from the player.'''
        distances = []
        for enemy in settings.enemies:
            distances.append(enemy.enemyDistance(player))
        return settings.enemies[distances.index(min(distances))]

    def createEnemies(self,player,type,steps):
        '''Creates the enemy of the game. The enemy created after a player move.'''
        if len(settings.enemies) < 20:
            if self.seeded:
                random.seed(self.seed + 6999 + steps*15)
            p0 = random.uniform(0.1, 0.25)
            p = 2*p0/math.exp(player.getMaxLevelHitpoints(player.getLevel())/player.hitPoints)
            # print(p)
            if p < 0.10:
                p = 0.10
            self.enemy = None
            if random.random() < p:
                pe = random.random()
                level = player.getLevel()
                
                if level == 1 and pe <= 0.6:
                    self.enemy = Giant_Rat()
                    
                if level == 1 and pe > 0.6:
                    self.enemy = Goblin()
                    
                if level == 2 and pe < 0.25:
                    self.enemy = Giant_Rat()
                    
                if level == 2 and pe >= 0.25 and pe < 0.75:
                    self.enemy = Goblin()
                    
                if level == 2 and pe > 0.75:
                    self.enemy = Gray_Slime()
                    
                if level == 3 and pe < 0.2:
                    self.enemy = Goblin()
                    
                if level == 3 and pe >= 0.2 and pe < 0.5:
                    self.enemy = Gray_Slime()
                    
                if level == 3 and pe >= 0.5 and pe < 0.8:
                    self.enemy = Orc_Grunt()
                    
                if level == 3 and pe >= 0.8 and pe < 0.9:
                    self.enemy = Orc_Warlord()
                    
                if level == 3 and pe >= 0.9:
                    self.enemy = Skeleton()
                    
                if level == 4 and pe < 0.2:
                    self.enemy = Orc_Grunt()
                    
                if level == 4 and pe >= 0.2 and pe < 0.4:
                    self.enemy = Ettin()
                    
                if level == 4 and pe >= 0.4 and pe < 0.7:
                    self.enemy = Skeleton()
                    
                if level == 4 and pe >= 0.7 and pe < 0.9:
                    self.enemy = Orc_Grunt()
                    
                if level == 4 and pe >= 0.9:
                    self.enemy = Skeleton()
                    
                if level == 5 and pe < 0.2:
                    self.enemy = Wyrm()
                    
                if level == 5 and pe >= 0.2 and pe < 0.4:
                    self.enemy = Ettin()
                    
                if level == 5 and pe >= 0.4 and pe < 0.7:
                    self.enemy = Orc_Warlord()
                    
                if level == 5 and pe >= 0.7 and pe < 0.9:
                    self.enemy = Vampire()
                    
                if level == 5 and pe >= 0.9:
                    self.enemy = Skeleton()
            if self.enemy != None:
                settings.enemies.append(self.enemy)
            dx = player.currentPositionX
            dy = player.currentPositionY
            enpx = 0
            enpy = 0
            enemy_position_rule = False

            if type == GameEnum.MovementType.left and self.enemy != None:
                for x in reversed(range(settings.xtile)):
                    for y in range(settings.ytile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]

                        
            if type == GameEnum.MovementType.right and self.enemy != None:
                for x in range(settings.xtile):
                    for y in range(settings.ytile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                               
            if type == GameEnum.MovementType.up and self.enemy != None:
                for y in reversed(range(settings.ytile)):
                    for x in range(settings.xtile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                         
            if type == GameEnum.MovementType.down and self.enemy != None:
                for y in range(settings.ytile):
                    for x in range(settings.xtile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                        

    def createEnemiesRest(self,player,rest):
        if len(settings.enemies) < 20:
            p = 0.45
            if self.seeded:
                random.seed(self.seed + 5999 + rest*5)
            self.enemy = None
            if random.random() < p:
                pe = random.random()
                level = player.getLevel()
                
                if level == 1 and pe <= 0.6:
                    self.enemy = Giant_Rat()
                    
                if level == 1 and pe > 0.6:
                    self.enemy = Goblin()
                    
                if level == 2 and pe < 0.25:
                    self.enemy = Giant_Rat()
                    
                if level == 2 and pe >= 0.25 and pe < 0.75:
                    self.enemy = Goblin()
                    
                if level == 2 and pe > 0.75:
                    self.enemy = Gray_Slime()
                    
                if level == 3 and pe < 0.2:
                    self.enemy = Goblin()
                    
                if level == 3 and pe >= 0.2 and pe < 0.5:
                    self.enemy = Gray_Slime()
                    
                if level == 3 and pe >= 0.5 and pe < 0.8:
                    self.enemy = Orc_Grunt()
                    
                if level == 3 and pe >= 0.8 and pe < 0.9:
                    self.enemy = Orc_Warlord()
                    
                if level == 3 and pe >= 0.9:
                    self.enemy = Skeleton()
                    
                if level == 4 and pe < 0.2:
                    self.enemy = Orc_Grunt()
                    
                if level == 4 and pe >= 0.2 and pe < 0.4:
                    self.enemy = Ettin()
                    
                if level == 4 and pe >= 0.4 and pe < 0.7:
                    self.enemy = Skeleton()
                    
                if level == 4 and pe >= 0.7 and pe < 0.9:
                    self.enemy = Orc_Grunt()
                    
                if level == 4 and pe >= 0.9:
                    self.enemy = Skeleton()
                    
                if level == 5 and pe < 0.2:
                    self.enemy = Wyrm()
                    
                if level == 5 and pe >= 0.2 and pe < 0.4:
                    self.enemy = Ettin()
                    
                if level == 5 and pe >= 0.4 and pe < 0.7:
                    self.enemy = Orc_Warlord()
                    
                if level == 5 and pe >= 0.7 and pe < 0.9:
                    self.enemy = Vampire()
                    
                if level == 5 and pe >= 0.9:
                    self.enemy = Skeleton()
                    
            case = random.randint(0, 3)
            if self.enemy != None:
                settings.enemies.append(self.enemy)
            dx = player.currentPositionX
            dy = player.currentPositionY
            enpx = 0
            enpy = 0
            enemy_position_rule = False
            if case == 0 and self.enemy != None:
                for x in reversed(range(settings.xtile)):
                    for y in range(settings.ytile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                        
            if case == 1 and self.enemy != None:
                for x in range(settings.xtile):
                    for y in range(settings.ytile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                               
            if case == 2 and self.enemy != None:
                for y in reversed(range(settings.ytile)):
                    for x in range(settings.xtile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True
                            

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                         
            if case == 3 and self.enemy != None:
                for y in range(settings.ytile):
                    for x in range(settings.xtile):
                        new_distance = abs(dx - x) + abs(dy - y)
                        if settings.tiles[x][y].ground == GameEnum.GroundType.floor and new_distance == settings.radius + self.enemy.visibility +1 and settings.tiles[x][y].occupancy !=True:
                            enpx = x 
                            enpy = y
                            enemy_position_rule = True

                if enemy_position_rule:            
                    self.enemy.enemyCurrentPossitionX = enpx
                    self.enemy.enemyCurrentPossitionY = enpy
                    settings.tiles[enpx][enpy].occupancy = True
                    player.playerHearing(self.enemy)
                if not enemy_position_rule:
                    del settings.enemies[-1]
                        


 
        

