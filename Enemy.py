from abc import ABC, abstractmethod
import GameSettings as settings
import GameEnum

class Enemy(ABC):
    '''The Enemy is an abstract class which is creating the enemies of the game.'''

    def __init__(self):
        '''Creates the Enemy object.'''
        self.hitPoints = 0
        self.strenghtPoints = 0
        self.XPreturn = 0
        self.visibility = 0
        self.name = None
        self.enemyCurrentPossitionX = 0
        self.enemyCurrentPossitionY = 0

    def enemyDistance(self,player):
        '''enemyDistance.Return the distance between player and enemy.
        The distance unit of measurement is tile.'''
        distance = abs(player.currentPositionX - self.enemyCurrentPossitionX) + abs(player.currentPositionY - self.enemyCurrentPossitionY)
        return distance

    def minDistance(self,player,x,y):
        '''minDistance.Returns the minimum distance that occur between player and enemy, 
        if the enemy moves to its adjacent tiles.
        The distance unit of measurement is tile.'''
        distances = []
        for i in range(x - 1,x + 2):
            for h in range(y - 1,y + 2):
                new_distance = abs(player.currentPositionX - i) + abs(player.currentPositionY - h)
                distances.append(new_distance)
        return min(distances)


    def enemyMovement(self,player):
        '''enemyMovement. Determines the movement of the enemy taking under consideration the distance between player and enemy.
        Basic aim is to reduce the distance, taking under consideration that the movement of the enemy is only on the x or y axis.'''
        distance = self.minDistance(player,self.enemyCurrentPossitionX,self.enemyCurrentPossitionY)
        new_distance_d,new_distance_l,new_distance_r,new_distance_u = 0,0,0,0
        if distance == 0:
            self.attack(player)
        else:
            distance = abs(player.currentPositionX - self.enemyCurrentPossitionX) + abs(player.currentPositionY - self.enemyCurrentPossitionY)
            if distance <= self.visibility:
                for type in GameEnum.MovementType:
                    if type == GameEnum.MovementType.down:
                        x = self.enemyCurrentPossitionX
                        y = self.enemyCurrentPossitionY + 1
                        new_distance_d = self.minDistance(player,x,y)
                        
                    if type == GameEnum.MovementType.up:
                        x = self.enemyCurrentPossitionX 
                        y = self.enemyCurrentPossitionY - 1
                        new_distance_u = self.minDistance(player,x,y)
                        
                    if type == GameEnum.MovementType.left:
                        x = self.enemyCurrentPossitionX - 1
                        y = self.enemyCurrentPossitionY
                        new_distance_l = self.minDistance(player,x,y)
                
                    if type == GameEnum.MovementType.right:
                        x = self.enemyCurrentPossitionX + 1
                        y = self.enemyCurrentPossitionY
                        new_distance_r = self.minDistance(player,x,y)
                x = 0
                y = 0
                
                if new_distance_d < distance and self.enemyCurrentPossitionY + 1 < 40 and settings.tiles[self.enemyCurrentPossitionX][self.enemyCurrentPossitionY+1].ground == GameEnum.GroundType.floor and settings.tiles[self.enemyCurrentPossitionX][self.enemyCurrentPossitionY + 1].occupancy == False:
                    x = self.enemyCurrentPossitionX  
                    y = self.enemyCurrentPossitionY+1
                    distance = new_distance_d

                if new_distance_u < distance and self.enemyCurrentPossitionY - 1 >= 0  and settings.tiles[self.enemyCurrentPossitionX][self.enemyCurrentPossitionY - 1].ground == GameEnum.GroundType.floor and settings.tiles[self.enemyCurrentPossitionX][self.enemyCurrentPossitionY - 1].occupancy == False:
                    x = self.enemyCurrentPossitionX  
                    y = self.enemyCurrentPossitionY - 1
                    distance = new_distance_u

                if new_distance_l < distance and self.enemyCurrentPossitionX - 1 >= 0 and settings.tiles[self.enemyCurrentPossitionX - 1][self.enemyCurrentPossitionY].ground == GameEnum.GroundType.floor and settings.tiles[self.enemyCurrentPossitionX - 1][self.enemyCurrentPossitionY].occupancy == False:
                    x = self.enemyCurrentPossitionX - 1 
                    y = self.enemyCurrentPossitionY
                    distance = new_distance_l

                if new_distance_r < distance and self.enemyCurrentPossitionX + 1 < 80 and settings.tiles[self.enemyCurrentPossitionX + 1][self.enemyCurrentPossitionY].ground == GameEnum.GroundType.floor and settings.tiles[self.enemyCurrentPossitionX + 1][self.enemyCurrentPossitionY].occupancy == False:
                    x = self.enemyCurrentPossitionX + 1 
                    y = self.enemyCurrentPossitionY
                    distance = new_distance_r
                    
                settings.tiles[self.enemyCurrentPossitionX][self.enemyCurrentPossitionY].occupancy = False
                self.enemyCurrentPossitionX = x
                self.enemyCurrentPossitionY = y
                settings.tiles[self.enemyCurrentPossitionX][self.enemyCurrentPossitionY].occupancy = True

                

    @abstractmethod
    def attack(self,player):
        '''attack. Performs the attack of the enemy against the player'''
        pass



    

