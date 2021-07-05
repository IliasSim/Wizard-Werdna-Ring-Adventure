'''The GameSettings is an singleton class which is used for the storage of parameters that many classes of the game have to have access.'''
import os
from Tile import Tile
xtile = 80
ytile = 40
mapWidth = 960
mapHeigth = 480
startX = 0
startY = 0
radius = 6
countfloortile = 0
exitx = 0
exity = 0
enemies = []
game_text = []

tiles = [[0]*ytile for i in range(xtile)]
for y in range(ytile):
    for x in range(xtile):
        tiles[x][y] = Tile(x,y)

def addGameText(text):
    '''Creates a list with the text to be depicted at the game'''
    if len(game_text) > 4:
        del game_text[0]    
    game_text.append(text)



