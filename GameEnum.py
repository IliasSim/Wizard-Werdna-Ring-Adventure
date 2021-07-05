from enum import Enum

class GroundType(Enum):
    '''GroundType is an enumerated type which
    defines the type of the ground the tile of the game has.'''
    wall = 1
    floor = 2
    stairs = 3

class VisibilityType(Enum):
    '''VisibilityType is an enumerated type which
    defines if a tile will be visible or not.'''
    unknown = 4
    fogged = 5
    visible = 6

class MovementType(Enum):
    '''MovementType is an enumerated type which
    defines the direction the player move.'''
    up = 7
    down = 8
    left = 9
    right = 10

class EffectType(Enum):
    '''EffectType is an enumerated type which
    defines if an item can cause effect.'''
    none = 11
    mana_boost = 12
    mana_replenish = 13
    hp_boost = 14
    hp_replenish = 15
    damage_boost = 16
    intelligence_boost = 17



