import GameEnum

class HelthPotion():

    def __init__(self):
        self.eq = False
        self.use = 1
        self.effectType1 = GameEnum.EffectType.hp_replenish
        self.hp_add = 20

class ManaPotion():

    def __init__(self):
        self.eq = False
        self.use = 1
        self.effectType1 = GameEnum.EffectType.mana_replenish
        self.mana_boost = 20

class Sword():

    def __init__(self,hpboost,damageboost,name):
        self.eq = False
        self.effectType1 = GameEnum.EffectType.hp_boost
        self.effectType2 = GameEnum.EffectType.damage_boost
        self.name = name
        self.hp_boost = hpboost
        self.damage_boost = damageboost

class Staff():

    def __init__(self,hpboost,manaboost,intelligence,name):
        self.eq = False
        self.effectType1 = GameEnum.EffectType.hp_boost
        self.effectType2 = GameEnum.EffectType.mana_boost
        self.effectType3 = GameEnum.EffectType.intelligence_boost
        self.name = name
        self.hp_boost = hpboost
        self.manaboost = manaboost
        self.intelligence_boost = intelligence

class Werdna_Ring():

    def __init__(self):
        pass