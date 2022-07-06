from re import X
import GameEnum

class HelthPotion():
    '''This class cretes an ithm for the game that can be consumed and adds 20 hp to the player hitpoint'''
    def __init__(self,x,y):
        self.eq = False
        self.use = 1
        self.effectType1 = GameEnum.EffectType.hp_replenish
        self.hp_add = 20
        self.x = x
        self.y = y
        self.seen = False

class ManaPotion():
    '''This class cretaes an item that can be used by the Wizard type of player and adds 20 manapoints to the player MP.'''
    def __init__(self,x,y):
        self.eq = False
        self.use = 1
        self.effectType1 = GameEnum.EffectType.mana_replenish
        self.mana_boost = 20
        self.x = x
        self.y = y
        self.seen = False

class Sword():

    def __init__(self,hpboost,damageboost,name,x,y):
        self.eq = False
        self.effectType1 = GameEnum.EffectType.hp_boost
        self.effectType2 = GameEnum.EffectType.damage_boost
        self.name = name
        self.hp_boost = hpboost
        self.damage_boost = damageboost
        self.x = x
        self.y = y
        self.seen = False

class Staff():

    def __init__(self,hpboost,manaboost,intelligence,name,x,y):
        self.eq = False
        self.effectType1 = GameEnum.EffectType.hp_boost
        self.effectType2 = GameEnum.EffectType.mana_boost
        self.effectType3 = GameEnum.EffectType.intelligence_boost
        self.name = name
        self.hp_boost = hpboost
        self.manaboost = manaboost
        self.intelligence_boost = intelligence
        self.x = x
        self.y = y
        self.seen = False

class Werdna_Ring():

    def __init__(self):
        pass