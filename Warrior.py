import GameSettings as settings
from AbstractPlayer import AbstractPlayer
from Games_items import HelthPotion,ManaPotion,Sword,Staff,Werdna_Ring

class Warrior(AbstractPlayer):
    '''The Warrior is a subclass of the AbstractPlayer class and 
    creates one of the player kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 30
        self.maxHitPoints = 30
        self.baseStrength = 10
        self.weapon = None
        self.level_table = dict(
            [(i,1) for i in range(0, 300)]
             + [(i,2) for i in range(300, 900)]
             +[(i,3) for i in range(900,2700)]
             +[(i,4) for i in range(2700,6500)]
             +[(i,5) for i in range(6500,14000)])
        

    # overriding abstract method
    def getLevel(self):
        return self.level_table[self.experiencePoints]

    # overriding abstract method
    def addXP(self,amount):
        currentLevel = self.getLevel()
        if self.experiencePoints >= 13999:
            pass
        else:
            self.experiencePoints = self.experiencePoints + amount
            if self.experiencePoints > 13999:
                self.experiencePoints = 13999
            if currentLevel < self.getLevel():
                self.addBonus(self.getLevel())
                text = self.name + " changes level. New level is "+ str(self.getLevel())
                settings.addGameText(text)
    
    def addBonus(self, level):
        if level == 2:
            self.maxHitPoints = self.maxHitPoints + 30
            self.baseStrength = self.baseStrength + 10
            return
        if level == 3:
            self.maxHitPoints = self.maxHitPoints + 20
            self.baseStrength = self.baseStrength + 5
            return
        if level == 4:
            self.maxHitPoints = self.maxHitPoints + 10
            self.baseStrength = self.baseStrength + 5
            return
        if level == 5:
            self.maxHitPoints = self.maxHitPoints + 10
            self.baseStrength = self.baseStrength + 5
            return


    def getManaFromEquipment(self):
        pass

    def getStrengthFromEquimpment(self):
        totalbaseStrength = 0
        if self.weapon != None:
            totalbaseStrength = self.weapon.damage_boost
        return totalbaseStrength

    def getHPFromEquipment(self):
        totalhitPoints = 0
        if self.weapon != None:
            totalhitPoints = self.weapon.hp_boost
        return totalhitPoints

    def use(self, item):
        oldhp = self.hitPoints
        if self.getLevel() == 1 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + item.hp_add
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
                
        if self.getLevel() == 2 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + item.hp_add
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
        if self.getLevel() == 3 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + item.hp_add
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
        if self.getLevel() == 4 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + item.hp_add
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
        if self.getLevel() == 5 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + item.hp_add
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
        text = self.name + " use a health potion and earn " + str(self.hitPoints - oldhp) + " HP."
        settings.addGameText(text)

            
        
    def getAttackDamage(self):
        return self.baseStrength + self.getStrengthFromEquimpment()
    
    def enemyToAttack(self):
        index = None
        hitpoints = 100
        if settings.enemies != []:
            i = 0
            for enemy in settings.enemies:
                i = i + 1
                if enemy.minDistance(self,enemy.enemyCurrentPossitionX,enemy.enemyCurrentPossitionY) == 0 and enemy.hitPoints <= hitpoints:
                    index = i - 1
                    hitpoints = enemy.hitPoints
        return index

    def pickItem(self):
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,HelthPotion): 
            self.pickUP(settings.tiles[self.currentPositionX][self.currentPositionY].store)
            settings.tiles[self.currentPositionX][self.currentPositionY].store = None
            text = self.name + ' found a health potion press h to use it.'
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,ManaPotion):
            text = self.name + " doesn't use Mana potion."
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,Sword):
            weapon = settings.tiles[self.currentPositionX][self.currentPositionY].store
            text = self.name + " press p to equip with the " + weapon.name + " which add " + str(weapon.hp_boost) + " to max HP and " + str(weapon.damage_boost) + " to player strength"
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,Staff):
            text = self.name + " can't use this type of weapon."
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,Werdna_Ring):
            text = self.name + " found Werdna's Ring press p to end game."
            settings.addGameText(text)

    def getWeapon(self):
        return self.weapon

    def setWeapon(self,weapon):
        self.weapon = weapon
        self.maxHitPoints = self.getMaxLevelHitpoints(self.getLevel()) + self.getHPFromEquipment()
        if self.getMaxLevelHitpoints(self.getLevel()) + self.getHPFromEquipment() < self.hitPoints:
            self.hitPoints = self.getMaxLevelHitpoints(self.getLevel()) + self.getHPFromEquipment()
    
    def rest(self):
        '''Adds 4 hp  to the warrior hp'''
        oldhp = self.hitPoints
        if self.getLevel() == 1 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + 4
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
                
        if self.getLevel() == 2 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + 4
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
        if self.getLevel() == 3 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + 4
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
        if self.getLevel() == 4 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + 4
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
                
        if self.getLevel() == 5 and self.hitPoints < self.maxHitPoints:
            self.hitPoints = self.hitPoints + 4
            if  self.hitPoints > self.maxHitPoints:
                self.hitPoints = self.maxHitPoints
        text = self.name + " rest and earn " + str(self.hitPoints - oldhp) + " HP."
        settings.addGameText(text)

    def getMaxLevelHitpoints(self, level):
        if level == 1:
            return 30
        if level == 2:
            return 60
        if level == 3:
            return 80
        if level == 4:
            return 90
        if level == 5:
            return 100
    
    def attack(self, enemy):
        attack = False
        if enemy.minDistance(self,enemy.enemyCurrentPossitionX,enemy.enemyCurrentPossitionY) == 0:
            enemy.hitPoints = enemy.hitPoints - self.getAttackDamage()
            attack = True
            text = self.name + " attack " + enemy.name + " for " + str(self.getAttackDamage()) + " HP damage."
            settings.addGameText(text)
        return attack 

    def getInteFromEquipment(self):
        pass

    def playerType(self):
        return 'Warrior'

    def dropWeapon(self):
        self.maxHitPoints = self.getMaxLevelHitpoints(self.getLevel()) - self.getHPFromEquipment()
        return self.weapon

    def getBaseIntelligence(self):
        pass

