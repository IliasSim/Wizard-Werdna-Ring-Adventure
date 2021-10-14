import GameSettings as settings
import GameEnum
import pygame
from Enemies import Vampire,Wyrm,Giant_Rat,Goblin,Gray_Slime,Ettin,Orc_Grunt,Orc_Warlord,Skeleton
from Games_items import HelthPotion,ManaPotion,Werdna_Ring,Staff,Sword

class GameMapGraphics():
    '''The GameMapGraphics class creates the image of game map. 
    Also it creates images of the enemies, the player and the items placed on the map.'''
    

    def __init__(self,screen):
        self.tilesize=4*settings.screenFactor
        self.tilefilsize=11/3*settings.screenFactor
        self.rectEnemyPading = 1/3*settings.screenFactor
        self.rectEnemyFill = 3*settings.screenFactor
        self.cycleEnemyPading = 5/3*settings.screenFactor
        self.cycleEnemyFill = 4/3*settings.screenFactor
        self.skeletonfill = 7/3*settings.screenFactor
        self.itemFill = 1*settings.screenFactor
        self.itemPading = 8/3*settings.screenFactor
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
                    pygame.draw.rect(self.screen, self.black, [x*self.tilesize,y*self.tilesize,self.tilefilsize,self.tilefilsize])
                if settings.tiles[x][y].ground == GameEnum.GroundType.floor:
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.unknown:
                        pygame.draw.rect(self.screen, self.gray, [x*self.tilesize,y*self.tilesize,self.tilefilsize,self.tilefilsize])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.visible:
                        pygame.draw.rect(self.screen, self.red, [x*self.tilesize,y*self.tilesize,self.tilefilsize,self.tilefilsize])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.fogged:
                        pygame.draw.rect(self.screen, self.darkred, [x*self.tilesize,y*self.tilesize,self.tilefilsize,self.tilefilsize])
                if settings.tiles[x][y].ground == GameEnum.GroundType.stairs:
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.unknown:
                        pygame.draw.rect(self.screen, self.black, [x*self.tilesize,y*self.tilesize,self.tilefilsize,self.tilefilsize])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.visible:
                        pygame.draw.rect(self.screen, self.yellow, [settings.exitx*self.tilesize,settings.exity*self.tilesize,self.tilefilsize,self.tilefilsize])
                    if  settings.tiles[x][y].visibility == GameEnum.VisibilityType.fogged:
                        pygame.draw.rect(self.screen, (255,200,0), [settings.exitx*self.tilesize,settings.exity*self.tilesize,self.tilefilsize,self.tilefilsize])

    def drawPlayer(self,x,y):
        '''Creates the image of the player.'''
        pygame.draw.rect(self.screen, self.blue, [x*self.tilesize + self.rectEnemyPading,y*self.tilesize + self.rectEnemyPading,self.rectEnemyFill,self.rectEnemyFill])
        

    def enemyDepiction(self):
        '''Creates the image of the enemies.'''
        if settings.exitx != []:
            for enemy in settings.enemies:
                if settings.tiles[enemy.enemyCurrentPossitionX][enemy.enemyCurrentPossitionY].visibility == GameEnum.VisibilityType.visible:
                    if isinstance(enemy, Giant_Rat):
                        pygame.draw.rect(self.screen, self.gray, [enemy.enemyCurrentPossitionX*self.tilesize + self.rectEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.rectEnemyPading,self.rectEnemyFill,self.rectEnemyFill])
                    if isinstance(enemy, Goblin):
                        pygame.draw.circle(self.screen, self.yellow, (enemy.enemyCurrentPossitionX*self.tilesize  + self.cycleEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.cycleEnemyPading), self.cycleEnemyFill)
                    if isinstance(enemy, Gray_Slime):
                        pygame.draw.rect(self.screen, self.lightGray, [enemy.enemyCurrentPossitionX*self.tilesize + self.rectEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.rectEnemyPading,self.rectEnemyFill,self.rectEnemyFill])
                    if isinstance(enemy, Orc_Grunt):
                        pygame.draw.rect(self.screen, self.green, [enemy.enemyCurrentPossitionX*self.tilesize + self.rectEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.rectEnemyPading,self.rectEnemyFill,self.rectEnemyFill])
                    if isinstance(enemy, Orc_Warlord):
                        pygame.draw.circle(self.screen, self.dark_green, (enemy.enemyCurrentPossitionX*self.tilesize  + self.cycleEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.cycleEnemyPading), self.cycleEnemyFill)
                    if isinstance(enemy, Ettin):
                        pygame.draw.circle(self.screen, self.dark_gray, (enemy.enemyCurrentPossitionX*self.tilesize  + self.cycleEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.cycleEnemyPading), self.cycleEnemyFill)
                    if isinstance(enemy, Skeleton):
                        pygame.draw.rect(self.screen, self.white, [enemy.enemyCurrentPossitionX*self.tilesize + self.rectEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.rectEnemyPading,self.rectEnemyFill,self.skeletonfill])  
                    if isinstance(enemy, Wyrm):
                        pygame.draw.rect(self.screen, self.magenta, [enemy.enemyCurrentPossitionX*self.tilesize + self.rectEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.rectEnemyPading,self.rectEnemyFill,self.rectEnemyFill]) 
                    if isinstance(enemy, Vampire):
                        pygame.draw.circle(self.screen, self.black, (enemy.enemyCurrentPossitionX*self.tilesize  + self.cycleEnemyPading,enemy.enemyCurrentPossitionY*self.tilesize + self.cycleEnemyPading), self.cycleEnemyFill)

    def drawItem(self):
        '''Creates the image of the items.'''
        for x in range(settings.xtile):
            for y in range(settings.ytile):
                if settings.tiles[x][y].visibility != GameEnum.VisibilityType.unknown:
                    if  isinstance(settings.tiles[x][y].store,HelthPotion):
                        pygame.draw.rect(self.screen, self.blue, [x*self.tilesize + self.itemPading,y*self.tilesize,self.itemFill,self.itemFill])
                    if isinstance(settings.tiles[x][y].store,ManaPotion):
                        pygame.draw.rect(self.screen, self.yellow, [x*self.tilesize + self.itemPading,y*self.tilesize,self.itemFill,self.itemFill])
                    if isinstance(settings.tiles[x][y].store,Staff) or isinstance(settings.tiles[x][y].store,Sword):
                        pygame.draw.rect(self.screen, self.green, [x*self.tilesize,y*self.tilesize,self.itemFill,self.itemFill])
                    if isinstance(settings.tiles[x][y].store,Werdna_Ring):
                        pygame.draw.rect(self.screen, self.green, [x*self.tilesize,y*self.tilesize,self.tilefilsize,self.tilefilsize])

    def arrayreturn(self):
        array = pygame.surfarray.array3d(self.screen)
        return array