import GameSettings as settings
import GameEnum
import pygame
from Enemies import Vampire,Wyrm,Giant_Rat,Goblin,Gray_Slime,Ettin,Orc_Grunt,Orc_Warlord,Skeleton
from Games_items import HelthPotion,ManaPotion,Werdna_Ring,Staff,Sword

class GameMapGraphics():
    '''The GameMapGraphics class creates the image of game map. 
    Also it creates images of the enemies, the player and the items placed on the map.'''

    def __init__(self,screen):
        self.screen = screen
        self.red = (255,0,0)
        self.darkred = (120,0,0)
        self.yellow = (255,255,0)
        self.blue = (0,0,255)
        self.gray = (127,127,127)
        self.orange = (255,100,10)
        self.lightGray = (211, 211, 211)
        self.green = (0,255,0)
        self.dark_green=(0, 100, 0)
        self.dark_gray = (169, 169, 169)
        self.white = (255,255,255)
        self.magenta = (255,0,230)
        self.black = (0,0,0)

    def drawMap(self):
        '''Creates the image of the game map.'''
        for x in range(settings.xtile):
            for y in range(settings.ytile):
                if settings.tiles[x][y].ground == GameEnum.GroundType.wall:
                    pygame.draw.rect(self.screen, self.black, [x*12,y*12,11,11])
                if settings.tiles[x][y].ground == GameEnum.GroundType.floor:
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.unknown:
                        pygame.draw.rect(self.screen, self.black, [x*12,y*12,11,11])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.visible:
                        pygame.draw.rect(self.screen, self.red, [x*12,y*12,11,11])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.fogged:
                        pygame.draw.rect(self.screen, self.darkred, [x*12,y*12,11,11])
                if settings.tiles[x][y].ground == GameEnum.GroundType.stairs:
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.unknown:
                        pygame.draw.rect(self.screen, self.black, [x*12,y*12,11,11])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.visible:
                        pygame.draw.rect(self.screen, self.yellow, [settings.exitx*12,settings.exity*12,11,11])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.fogged:
                        pygame.draw.rect(self.screen, (255,200,0), [settings.exitx*12,settings.exity*12,11,11])

    def drawPlayer(self,x,y):
        '''Creates the image of the player.'''
        pygame.draw.rect(self.screen, self.blue, [x*12 + 1,y*12 + 1,9,9])
        

    def enemyDepiction(self):
        '''Creates the image of the enemies.'''
        if settings.exitx != []:
            for enemy in settings.enemies:
                if settings.tiles[enemy.enemyCurrentPossitionX][enemy.enemyCurrentPossitionY].visibility == GameEnum.VisibilityType.visible:
                    if isinstance(enemy, Giant_Rat):
                        pygame.draw.rect(self.screen, self.gray, [enemy.enemyCurrentPossitionX*12 + 1,enemy.enemyCurrentPossitionY*12 + 1,9,9])
                    if isinstance(enemy, Goblin):
                        pygame.draw.circle(self.screen, self.yellow, (enemy.enemyCurrentPossitionX*12  + 5,enemy.enemyCurrentPossitionY*12 + 5), 4)
                    if isinstance(enemy, Gray_Slime):
                        pygame.draw.rect(self.screen, self.lightGray, [enemy.enemyCurrentPossitionX*12 + 1,enemy.enemyCurrentPossitionY*12 + 1,9,9])
                    if isinstance(enemy, Orc_Grunt):
                        pygame.draw.rect(self.screen, self.green, [enemy.enemyCurrentPossitionX*12 + 1,enemy.enemyCurrentPossitionY*12 + 1,9,9])
                    if isinstance(enemy, Orc_Warlord):
                        pygame.draw.circle(self.screen, self.dark_green, (enemy.enemyCurrentPossitionX*12  + 5,enemy.enemyCurrentPossitionY*12 + 5), 4)
                    if isinstance(enemy, Ettin):
                        pygame.draw.circle(self.screen, self.dark_gray, (enemy.enemyCurrentPossitionX*12  + 5,enemy.enemyCurrentPossitionY*12 + 5), 4)
                    if isinstance(enemy, Skeleton):
                        pygame.draw.rect(self.screen, self.white, [enemy.enemyCurrentPossitionX*12 + 1,enemy.enemyCurrentPossitionY*12 + 1,9,7])  
                    if isinstance(enemy, Wyrm):
                        pygame.draw.rect(self.screen, self.magenta, [enemy.enemyCurrentPossitionX*12 + 1,enemy.enemyCurrentPossitionY*12 + 1,9,9]) 
                    if isinstance(enemy, Vampire):
                        pygame.draw.circle(self.screen, self.black, (enemy.enemyCurrentPossitionX*12  + 5,enemy.enemyCurrentPossitionY*12 + 5), 4)

    def drawItem(self):
        '''Creates the image of the items.'''
        for x in range(settings.xtile):
            for y in range(settings.ytile):
                if settings.tiles[x][y].visibility != GameEnum.VisibilityType.unknown:
                    if  isinstance(settings.tiles[x][y].store,HelthPotion):
                        pygame.draw.rect(self.screen, self.blue, [x*12 + 8,y*12,3,3])
                    if isinstance(settings.tiles[x][y].store,ManaPotion):
                        pygame.draw.rect(self.screen, self.yellow, [x*12 + 8,y*12,3,3])
                    if isinstance(settings.tiles[x][y].store,Staff) or isinstance(settings.tiles[x][y].store,Sword):
                        pygame.draw.rect(self.screen, self.green, [x*12,y*12,3,3])
                    if isinstance(settings.tiles[x][y].store,Werdna_Ring):
                        pygame.draw.rect(self.screen, self.green, [x*12,y*12,11,11])