from GameEnum import VisibilityType,GroundType

class Tile():
    '''The Tile class represent one place of the map. 
    In this game each tile occupy 12 * 12 pixels of the computer screen.'''

    def __init__(self,x,y):
        self.ground = GroundType.wall
        self.visibility = VisibilityType.unknown
        self.position_y = y
        self.position_x = x
        self.occupancy = False
        self.store = None

