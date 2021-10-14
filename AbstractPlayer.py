from abc import ABC, abstractmethod
import GameSettings as settings
import GameEnum

class AbstractPlayer(ABC):
    '''The AbstractPlayer is an abstract class which is creating the player of the game.'''
    def __init__(self):
        self.hitPoints = 0
        self.maxHitPoints = 0
        self.experiencePoints = 0
        self.inventory = []
        self.name = None
        self.currentPositionX = settings.startX
        self.currentPositionY = settings.startY

    @abstractmethod
    def addXP(self,amount):
        '''addXP adds XP to the player'''
        pass

    @abstractmethod
    def getLevel():
        '''getLevel Returns the current level of the player'''
        pass

    @abstractmethod
    def addBonus(self,level):
        '''addBonus add extra points hp, mana, strength and intelligence to the player when he change level.'''
        pass

    @abstractmethod
    def getManaFromEquipment(self):
        '''getManaFromEquipment returns the mana points equipped weapon add to the player.'''
        pass

    @abstractmethod
    def getStrengthFromEquimpment(self):
        '''getStrengthFromEquipment returns the strength points equipped weapon add to the player.'''
        pass

    @abstractmethod
    def getHPFromEquipment(self):
        '''getHPFromEquipment returns the health points equipped weapon add to the player.'''
        pass
    
    @abstractmethod
    def getBaseIntelligence(self):
        '''getBaseIntelligence returns the basic intelligence points the player possess.'''
        pass

    
    def pickUP(self,item):
        '''pickUP adds item to the player inventory.'''
        self.inventory.append(item)
    
    @abstractmethod
    def use(self,item):
        '''use.Uses an item from the player inventory.'''
    pass
        
    @abstractmethod
    def getAttackDamage(self):
        '''getAttackDamage. Gives the damage the player can cause.'''
        pass

    @abstractmethod
    def enemyToAttack(self):
        '''enemyToAttack. Chooses which enemy to attack.'''
        pass

    def setCurrentPositionX(self,x):
        '''setCurrentPositionX.Sets the position of the player at X axis.'''
        self.currentPositionX = x

    def setCurrentPositionY(self,y):
        '''setCurrentPositionY.Sets the position of the player at Y axis.'''
        self.currentPositionY = y

    def playerVisibility(self,x,y):
        ''' playerVisibility.Defines the visibility of the player.'''
        for i in range(settings.xtile):
            for h in range(settings.ytile):
                distance = abs(x - i) + abs(y - h)
                if distance <= settings.radius:
                    settings.tiles[i][h].visibility = GameEnum.VisibilityType.visible
                if distance > settings.radius and settings.tiles[i][h].visibility == GameEnum.VisibilityType.visible:
                    settings.tiles[i][h].visibility = GameEnum.VisibilityType.fogged

    def playerMovement(self,move):
        '''playerMovement.Determines in which direction the player can move.'''
        if move == GameEnum.MovementType.left:
            if self.currentPositionX-1 < 0:
                return
            if settings.tiles[self.currentPositionX - 1][self.currentPositionY].ground == GameEnum.GroundType.floor or settings.tiles[self.currentPositionX - 1][self.currentPositionY].ground == GameEnum.GroundType.stairs:
                if settings.tiles [self.currentPositionX - 1] [self.currentPositionY].occupancy != True:
                    settings.tiles [self.currentPositionX] [self.currentPositionY].occupancy = False
                    settings.tiles [self.currentPositionX - 1] [self.currentPositionY].occupancy = True
                    self.currentPositionX = self.currentPositionX - 1

        if move == GameEnum.MovementType.right:
            if self.currentPositionX + 1 > settings.xtile - 1:
                return
            if settings.tiles[self.currentPositionX + 1][self.currentPositionY].ground == GameEnum.GroundType.floor or settings.tiles[self.currentPositionX + 1][self.currentPositionY].ground == GameEnum.GroundType.stairs:
                if settings.tiles [self.currentPositionX + 1] [self.currentPositionY].occupancy != True:
                    settings.tiles [self.currentPositionX] [self.currentPositionY].occupancy = False
                    settings.tiles [self.currentPositionX + 1] [self.currentPositionY].occupancy = True
                    self.currentPositionX = self.currentPositionX + 1

        if move == GameEnum.MovementType.up:
            if self.currentPositionY-1 < 0:
                return
            if settings.tiles[self.currentPositionX][self.currentPositionY - 1].ground == GameEnum.GroundType.floor or settings.tiles[self.currentPositionX][self.currentPositionY - 1].ground == GameEnum.GroundType.stairs:
                if settings.tiles [self.currentPositionX] [self.currentPositionY - 1].occupancy != True:
                    settings.tiles [self.currentPositionX] [self.currentPositionY].occupancy = False
                    settings.tiles [self.currentPositionX] [self.currentPositionY - 1].occupancy = True
                    self.currentPositionY = self.currentPositionY - 1

        if move == GameEnum.MovementType.down:
            if self.currentPositionY + 1 > settings.ytile -1:
                return
            if settings.tiles[self.currentPositionX][self.currentPositionY + 1].ground == GameEnum.GroundType.floor or settings.tiles[self.currentPositionX][self.currentPositionY + 1].ground == GameEnum.GroundType.stairs:
                if settings.tiles [self.currentPositionX] [self.currentPositionY + 1].occupancy != True:
                    settings.tiles [self.currentPositionX] [self.currentPositionY].occupancy = False
                    settings.tiles [self.currentPositionX] [self.currentPositionY + 1].occupancy = True
                    self.currentPositionY = self.currentPositionY + 1
        self.pickItem()
        self.playerVisibility(self.currentPositionX, self.currentPositionY)

    @abstractmethod
    def pickItem(self):
        '''pickItem.Checks if the tile the player occupy has an item in its store and depending the class of the item'''
        pass

    def setFirstPosition(self,x,y):
        '''setFirstPosition.Sets the first position of the player in a map.'''
        settings.startX = x
        settings.startY = y
        self.currentPositionX = settings.startX
        self.currentPositionY = settings.startY

    @abstractmethod
    def rest(self):
        pass

    @abstractmethod
    def getMaxLevelHitpoints(self,level):
        '''getMaxLevelHitpoints.Gets the maximum HP a player can have based on it's level.'''

    @abstractmethod
    def attack(self,enemy):
        '''attack.Performs the attack of the player.'''
        pass

    @abstractmethod
    def getInteFromEquipment(self):
        '''getInteFromEquipment.Returns the intelligence the player has earn from his weapon.'''
        pass

    @abstractmethod
    def playerType(self):
        '''Returns the type of the player.'''
        pass

    @abstractmethod
    def getWeapon(self):    
        '''getWeapon.Returns the weapon the player possess.'''
        pass
    
    @abstractmethod
    def setWeapon(self,weapon):
        '''setWeapon.Sets the weapon the player possess and adds the effects the weapon offers.'''
        pass

    @abstractmethod
    def dropWeapon(self):
        '''dropWeapon.Returns the weapon the player possess and subtracts the effect the weapon causes.'''
        pass

    def playerHearing(self,enemy):
        '''Prints on game log the position a new enemy is created.'''
        
        if enemy.enemyCurrentPossitionX > self.currentPositionX and enemy.enemyCurrentPossitionY == self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name +  ' from the east.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX < self.currentPositionX and enemy.enemyCurrentPossitionY == self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the west.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX == self.currentPositionX and enemy.enemyCurrentPossitionY < self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the north.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX == self.currentPositionX and enemy.enemyCurrentPossitionY > self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the south.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX > self.currentPositionX and enemy.enemyCurrentPossitionY > self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the south east.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX > self.currentPositionX and enemy.enemyCurrentPossitionY < self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the north east.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX < self.currentPositionX and enemy.enemyCurrentPossitionY > self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the south west.'
            settings.addGameText(text)
        if enemy.enemyCurrentPossitionX < self.currentPositionX and enemy.enemyCurrentPossitionY < self.currentPositionY:
            text = self.name + ' can hear ' + enemy.name + ' from the north west.'
            settings.addGameText(text)
            
        