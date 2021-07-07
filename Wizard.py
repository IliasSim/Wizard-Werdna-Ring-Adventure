import GameSettings as settings
from AbstractPlayer import AbstractPlayer
from Games_items import HelthPotion,ManaPotion,Sword,Staff,Werdna_Ring

class Wizard(AbstractPlayer):
    '''The Wizard is a subclass of the AbstractPlayer class and 
    creates one of the player kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 20
        self.maxHitPoints = 20
        self.baseIntelligence = 10
        self.manaPoints = 30
        self.maxManaPoints = 30
        self.weapon = None
        self.level_table = dict(
            [(i,1) for i in range(0, 299)]
             + [(i,2) for i in range(300, 899)]
             +[(i,3) for i in range(900,2699)]
             +[(i,4) for i in range(2700,6499)]
             +[(i,5) for i in range(6500,14000)])
        

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
            self.maxHitPoints = self.maxHitPoints + 20
            self.baseIntelligence = self.baseIntelligence +10
            self.maxManaPoints = self.maxManaPoints + 20
            return
        if level == 3:
            self.maxHitPoints = self.maxHitPoints + 10
            self.baseIntelligence = self.baseIntelligence + 10
            self.maxManaPoints = self.maxManaPoints + 20
            return
        if level == 4:
            self.maxHitPoints = self.maxHitPoints + 5
            self.baseIntelligence = self.baseIntelligence + 10
            self.maxManaPoints = self.maxManaPoints + 20
            return
        if level == 5:
            self.maxHitPoints = self.maxHitPoints + 5
            self.baseIntelligence = self.baseIntelligence + 10
            self.maxManaPoints = self.maxManaPoints + 20
            return

    def getManaFromEquipment(self):
        totalManaPoints = 0
        if self.weapon != None:
            totalManaPoints = self.weapon.manaboost
        return totalManaPoints

    def getStrengthFromEquimpment(self):
        pass

    def getHPFromEquipment(self):
        totalhitPoints = 0
        if self.weapon != None:
            totalhitPoints = self.weapon.hp_boost
        return totalhitPoints
    
    def getBaseIntelligence(self):
        return self.baseIntelligence

    def use(self, item):
        if isinstance(item,HelthPotion):
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

        if isinstance(item,ManaPotion):
            oldmp = self.manaPoints
            if self.getLevel() == 1 and self.manaPoints < self.maxManaPoints:
                self.manaPoints = self.manaPoints + item.mana_boost
                if  self.manaPoints > self.maxManaPoints:
                    self.manaPoints = self.maxManaPoints
                
            if self.getLevel() == 2 and self.manaPoints < self.maxManaPoints:
                self.manaPoints = self.manaPoints + item.mana_boost
                if  self.manaPoints > self.maxManaPoints:
                    self.manaPoints = self.maxManaPoints
                
            if self.getLevel() == 3 and self.manaPoints < self.maxManaPoints:
                self.manaPoints = self.manaPoints + item.mana_boost
                if  self.manaPoints > self.maxManaPoints:
                    self.manaPoints = self.maxManaPoints
                
            if self.getLevel() == 4 and self.manaPoints < self.maxManaPoints:
                self.manaPoints = self.manaPoints + item.mana_boost
                if  self.manaPoints > self.maxManaPoints:
                    self.manaPoints = self.maxManaPoints
                
            if self.getLevel() == 5 and self.manaPoints < self.maxManaPoints:
                self.manaPoints = self.manaPoints + item.mana_boost
                if  self.manaPoints > self.maxManaPoints:
                    self.manaPoints = self.maxManaPoints
            text = self.name + " use a mana potion and earn " + str(self.manaPoints - oldmp) + " MP."
            settings.addGameText(text)

    def getAttackDamage(self):
        return self.baseIntelligence + self.getInteFromEquipment()
    
    def enemyToAttack(self):
        index = None
        hitpoints = 100
        if settings.enemies != []:
            i = 0
            distance = 6
            for enemy in settings.enemies:
                i = i + 1
                if enemy.minDistance(self,enemy.enemyCurrentPossitionX,enemy.enemyCurrentPossitionY) < distance and enemy.hitPoints <= hitpoints:
                    index = i - 1
                    distance = enemy.minDistance(self,enemy.enemyCurrentPossitionX,enemy.enemyCurrentPossitionY)
                    hitpoints = enemy.hitPoints
        return index

    def pickItem(self):
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,HelthPotion): 
            self.pickUP(settings.tiles[self.currentPositionX][self.currentPositionY].store)
            settings.tiles[self.currentPositionX][self.currentPositionY].store = None
            text = self.name + ' found a health potion press h to use it.'
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,ManaPotion):
            self.pickUP(settings.tiles[self.currentPositionX][self.currentPositionY].store)
            settings.tiles[self.currentPositionX][self.currentPositionY].store = None
            text = self.name + 'found a Mana potion press m to use it.'
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,Sword):
            text = self.name + " can't use this type of weapon."
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,Staff):
            weapon = settings.tiles[self.currentPositionX][self.currentPositionY].store
            text = "Press p to equip with the " + weapon.name + " which add " + str(weapon.hp_boost) + " to max HP and " + str(weapon.manaboost) + " to max MP and " + str(weapon.intelligence_boost) + ' to player inteligence'
            settings.addGameText(text)
        if settings.tiles[self.currentPositionX][self.currentPositionY].store != None and isinstance(settings.tiles[self.currentPositionX][self.currentPositionY].store,Werdna_Ring):
            text = self.name + " found Werdna's Ring press p to end game."
            settings.addGameText(text)

    def rest(self):
        '''Adds 4 hp and 4 mp  to the warrior total hp and total mp.'''
        oldhp = self.hitPoints
        oldmp = self.manaPoints

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
            
        if self.getLevel() == 1 and self.manaPoints < self.maxManaPoints:
            self.manaPoints = self.manaPoints + 4
            if  self.manaPoints > self.maxManaPoints:
                self.manaPoints = self.maxManaPoints
            
        if self.getLevel() == 2 and self.manaPoints < self.maxManaPoints:
            self.manaPoints = self.manaPoints + 4
            if  self.manaPoints > self.maxManaPoints:
                self.manaPoints = self.maxManaPoints
            
        if self.getLevel() == 3 and self.manaPoints < self.maxManaPoints:
            self.manaPoints = self.manaPoints + 4
            if  self.manaPoints > self.maxManaPoints:
                self.manaPoints = self.maxManaPoints
            
        if self.getLevel() == 4 and self.manaPoints < self.maxManaPoints:
            self.manaPoints = self.manaPoints + 4
            if  self.manaPoints > self.maxManaPoints:
                self.manaPoints = self.maxManaPoints
            
        if self.getLevel() == 5 and self.manaPoints < self.maxManaPoints:
            self.manaPoints = self.manaPoints + 4
            if  self.manaPoints > self.maxManaPoints:
                self.manaPoints = self.maxManaPoints
        text = self.name + " rest and earn " + str(self.hitPoints - oldhp) + " HP" + str(self.manaPoints - oldmp) + " MP"
        settings.addGameText(text)
        
    def attack(self, enemy):
        attack = False
        if enemy.enemyDistance(self) <= settings.radius and self.manaPoints >= 5:
            self.manaPoints = self.manaPoints - 5
            enemy.hitPoints = enemy.hitPoints - self.getAttackDamage()
            attack = True
            text = self.name + " attack " + enemy.name + " for " + str(self.getAttackDamage()) + " HP damage."
            settings.addGameText(text)
        return attack

    def getInteFromEquipment(self):
        totalIntelPoints = 0
        if self.weapon != None:
            totalIntelPoints = self.weapon.intelligence_boost
        return totalIntelPoints
    
    def playerType(self):
        return 'Wizard'

    def getWeapon(self):
        return self.weapon

    def getMaxLevelManapoints(self,level):
        '''Returns the maximum MP the Wizard can have based on it's level.'''
        if level == 1:
            return 30
        if level == 2:
            return 50
        if level == 3:
            return 70
        if level == 4:
            return 90
        if level == 5:
            return 110

    def getMaxLevelHitpoints(self, level):
        if level == 1:
            return 20
        if level == 2:
            return 40
        if level == 3:
            return 50
        if level == 4:
            return 55
        if level == 5:
            return 60


    def setWeapon(self,weapon):
        self.weapon = weapon
        self.maxHitPoints = self.getMaxLevelHitpoints(self.getLevel()) + self.getHPFromEquipment()
        if self.getMaxLevelHitpoints(self.getLevel()) + self.getHPFromEquipment() < self.hitPoints:
            self.hitPoints = self.getMaxLevelHitpoints(self.getLevel()) + self.getHPFromEquipment()
        self.maxManaPoints = self.getMaxLevelManapoints(self.getLevel()) + self.getManaFromEquipment()
        if self.getMaxLevelManapoints(self.getLevel()) + self.getManaFromEquipment() < self.manaPoints:
            self.manaPoints = self.getMaxLevelManapoints(self.getLevel()) + self.getManaFromEquipment()

    def dropWeapon(self):
        self.maxHitPoints = self.getMaxLevelHitpoints(self.getLevel()) - self.getHPFromEquipment()
        self.maxManaPoints = self.getMaxLevelManapoints(self.getLevel()) - self.getManaFromEquipment()
        return self.weapon

