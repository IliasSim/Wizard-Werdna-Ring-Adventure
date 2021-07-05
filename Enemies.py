from Enemy import Enemy
import GameSettings as settings
from Warrior import Warrior

class Ettin(Enemy):
    '''The Ettin is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 60
        self.strenghtPoints = 20
        self.XPreturn = 150
        self.visibility = 9
        self.name = "an Ettin"
        
    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)
        
class Giant_Rat(Enemy):
    '''The Giant_Rat is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 5
        self.strenghtPoints = 2
        self.XPreturn = 30
        self.visibility = 4
        self.name = "a Giant Rat"
        
    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Goblin(Enemy):
    '''The Goblin is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 15
        self.strenghtPoints = 5
        self.XPreturn = 50
        self.visibility = 7
        self.name = "a Goblin"
        
    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Gray_Slime(Enemy):
    '''The Gray_Slime is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 30
        self.strenghtPoints = 8
        self.XPreturn = 80
        self.visibility = 2
        self.name = "a Gray Slime"
        
    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Orc_Grunt(Enemy):
    '''The Orc_Grunt is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 40
        self.strenghtPoints = 10
        self.XPreturn = 100
        self.visibility = 6
        self.name = "an  Orc Grunt"
        

    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Orc_Warlord(Enemy):
    '''The Orc_Warlord is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 50
        self.strenghtPoints = 12
        self.XPreturn = 120
        self.visibility = 7
        self.name = "an Orc Warlord"
        

    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Skeleton(Enemy):
    '''The Skeleton is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 20
        self.strenghtPoints = 30
        self.XPreturn = 100
        self.visibility = 4
        self.name = "a Skeleton"
        

    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Vampire(Enemy):
    '''The Vampire is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 50
        self.strenghtPoints = 30
        self.XPreturn = 400
        self.visibility = 10
        self.name = "a Vampire"
        

    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)

class Wyrm(Enemy):
    '''The Wyrm is a subclass of the Enemy class and creates one of the enemy kinds.'''
    def __init__(self):
        super() .__init__()
        self.hitPoints = 80
        self.strenghtPoints = 20
        self.XPreturn = 200
        self.visibility = 5
        self.name = "a Wyrm"
        

    # overriding abstract method
    def attack(self, player):
        player.hitPoints = player.hitPoints - self.strenghtPoints
        text = player.name + " attacked by " +self.name + " for " + str(self.strenghtPoints) + " damage."
        settings.addGameText(text)
        

    

