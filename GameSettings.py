'''The GameSettings is a module  which is used for the storage of parameters that many classes of the game have to have access.'''

screenFactor = 3
reward = 0
gameCount = 0
mapWidth = 0
mapHeigth = 0
xtile = 0
ytile = 0
startX = 0
startY = 0
radius = 6
countfloortile = 0
exitx = 0
exity = 0
enemies = []
game_text = []
tiles = []

def addGameText(text):
    '''Creates a list with the text to be depicted at the game'''
    if len(game_text) > 4:
        del game_text[0]    
    game_text.append(text)



