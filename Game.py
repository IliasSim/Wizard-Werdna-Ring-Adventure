import sys
import GameSettings as settings
from GameMap import GameMap
import pygame
from GameMapGraphics import GameMapGraphics
from Games_items import HelthPotion,ManaPotion,Staff,Sword,Werdna_Ring
from Warrior import Warrior
from Wizard import Wizard
from Enemy import Enemy
import GameEnum
import random

def gameOver():
    '''Checks if the player is dead.'''
    if player.hitPoints <= 0:
        print(player.name,'is dead. Game Over')
        sys.exit()

def makeMap(cave):
    '''Creates the map of the game.'''
    precentage = 0
    makemap = True 
    while makemap:
        precentage = random.random()
        if 0.35 < precentage <= 0.55:
            makemap = False
    if cave >= 1:
        map.refreshTilesSettings()
    map.MakeMAp(precentage, player, cave)
    player.playerVisibility(player.currentPositionX, player.currentPositionY)

def countHealthPotion():
    '''Counts the health potions player posses.'''
    count = 0
    if player.inventory != []:
        for item in player.inventory:
            if isinstance(item, HelthPotion):
                count = count + 1
    return count

def countManaPotion():
    '''Counts the health potions player posses.'''
    count = 0
    if player.inventory != []:
        for item in player.inventory:
            if isinstance(item,ManaPotion):
                count = count + 1
    return count

def printPlayerStatus():
    '''Prints the player name and status'''
    font = pygame.font.Font('freesansbold.ttf', 12)
    if isinstance(player, Warrior):
        text = font.render('Warrior', True, (255,0,0), (0,0,0))
    if isinstance(player, Wizard):
        text = font.render('Wizard', True, (255,0,0), (0,0,0))
    text1 = font.render(player.name, True, (255,0,0), (0,0,0))
    text2 = font.render("HP" + str(player.hitPoints) + " /" + str(player.maxHitPoints), True, (255,0,0), (0,0,0)) 
    if isinstance(player, Warrior):
        text25 = font.render("MP 0/0" , True, (255,0,0), (0,0,0))    
    if isinstance(player, Wizard):
        text25 = font.render("MP" + str(player.manaPoints) + "/" + str(player.maxManaPoints) , True, (255,0,0), (0,0,0))  
    if player.weapon == None:
        text3 = font.render("Weapon: None " , True, (255,0,0), (0,0,0))
    else:
        text3 = font.render("Weapon: " + player.weapon.name , True, (255,0,0), (0,0,0))
    if isinstance(player, Warrior):
        text4 = font.render('Strength: ' + str(player.baseStrength), True, (255,0,0), (0,0,0))
    if isinstance(player, Wizard):
        text4 = font.render('Intelligence: ' + str(player.getBaseIntelligence()), True, (255,0,0), (0,0,0))
    text5 = font.render('Damage: ' + str(player.getAttackDamage()), True, (255,0,0), (0,0,0))
    text6 = font.render('Potions: ' + str(countHealthPotion())+ "(H) /" + str(countManaPotion()) + "(M)", True, (255,0,0), (0,0,0))
    text7 = font.render('Level: ' + str(player.getLevel()), True, (255,0,0), (0,0,0))
    text8 = font.render('XP: ' + str(player.experiencePoints), True, (255,0,0), (0,0,0))   
    screen.blit(text, (settings.mapWidth,8))
    screen.blit(text1, (settings.mapWidth,21))
    screen.blit(text2, (settings.mapWidth,34))
    screen.blit(text25, (settings.mapWidth,47))
    screen.blit(text3, (settings.mapWidth,60))
    screen.blit(text4, (settings.mapWidth,73))
    screen.blit(text5, (settings.mapWidth,86))
    screen.blit(text6, (settings.mapWidth,99))
    screen.blit(text7, (settings.mapWidth,112))
    screen.blit(text8, (settings.mapWidth,125))

def printText():
    '''Prints the game Log on the screen'''
    font = pygame.font.Font('freesansbold.ttf', 12)
    screen.fill(pygame.Color("black"))
    for i in range(len(settings.game_text)):
        text = font.render(settings.game_text[i], True, (255,0,0), (0,0,0))
        screen.blit(text,(0,settings.mapHeigth+(13*i)))
    
def enemyMove():
    '''Determines the movement of the enemy after the movement of the player.'''
    if settings.enemies != []:
            for enemy in settings.enemies:
                enemy.enemyMovement(player)
    gameOver()

def run_game():
    '''Initialize game and create a screen object.'''
    global map 
    map = GameMap()
    pygame.init()
    global screen
    screen = pygame.display.set_mode((settings.mapWidth+150, settings.mapHeigth+70))
    font = pygame.font.Font('freesansbold.ttf', 12)
    text = font.render("Please select the type of the player (Warrior or Wizard). Press 1 for Warrior or 2 for Wizard.", True, (255,0,0), (0,0,0))
    textRect = text.get_rect()
    textRect.center = ((settings.mapWidth+150)/2,(settings.mapHeigth+70)/2)
    screen.blit(text, textRect)
    pygame.display.set_caption("Wizard Werdna Ring")
    pygame.display.flip()
    writeText = True
    while writeText:
        typeinput = True
        for event in pygame.event.get():
            global player

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
            
                if event.key == pygame.K_1:
                    player = Warrior()
                    writeText = False
                if event.key == pygame.K_2:
                    player = Wizard()
                    writeText = False   
                else :
                    pass

    text = font.render("Please enter the name of the player and press enter: ", True, (255,0,0), (0,0,0))
    textRect = text.get_rect()
    input_rect = pygame.Rect((settings.mapWidth+150)/2-100, (settings.mapHeigth+70)/2 + 20, 200, 20)
    textRect.center = ((settings.mapWidth+150)/2,(settings.mapHeigth+70)/2)
    screen.fill(pygame.Color("black"))
    user_text = ''
    screen.blit(text, textRect)
    pygame.display.set_caption("Wizard Werdna Ring")
    pygame.display.flip()
    writeText = True
    while writeText:
        nameinput = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    while nameinput:
                        player.name = user_text
                        nameinput = False
                        writeText = False           

                if event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
  
                if event.key != pygame.K_BACKSPACE and event.key != pygame.K_RETURN:
                    user_text += event.unicode
        pygame.draw.rect(screen, (255,255,0), input_rect)
        text_surface = font.render(user_text, True, (255, 0, 0))

        screen.blit(text_surface, (input_rect.x+5, input_rect.y+5))
      
        input_rect.w = max(100, text_surface.get_width()+10)
        pygame.display.flip()


    cave = 0
    text = 'Welcome to the Wizard Werdna Ring adventure. Try to find the ring and win the Game.'
    settings.addGameText(text)
    printText()
    makeMap(cave)
    printPlayerStatus()
    DrawMap = GameMapGraphics(screen)
    DrawMap.drawMap()
    DrawMap.drawPlayer(player.currentPositionX, player.currentPositionY)
    DrawMap.drawItem()
    pygame.display.set_caption("Wizard Werdna Ring")
    pygame.display.flip()
    

    # Start the main loop for the game.
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                Xposition = player.currentPositionX
                Yposition = player.currentPositionY
                if event.key == pygame.K_a:
                    player.playerMovement(GameEnum.MovementType.left)
                    if Xposition == player.currentPositionX and Yposition == player.currentPositionY:
                        pass
                    else :
                        map.createEnemies(player, GameEnum.MovementType.left)
                        enemyMove()  
                    
                    
                if event.key == pygame.K_w:
                    player.playerMovement(GameEnum.MovementType.up)
                    if Xposition == player.currentPositionX and Yposition == player.currentPositionY:
                        pass
                    else :
                        map.createEnemies(player, GameEnum.MovementType.up)
                        enemyMove()

                if event.key == pygame.K_d:
                    player.playerMovement(GameEnum.MovementType.right)
                    if Xposition == player.currentPositionX and Yposition == player.currentPositionY:
                        pass
                    else :
                        map.createEnemies(player, GameEnum.MovementType.right)
                        enemyMove()  

                if event.key == pygame.K_s:
                    player.playerMovement(GameEnum.MovementType.down)
                    if Xposition == player.currentPositionX and Yposition == player.currentPositionY:
                        pass
                    else :
                        map.createEnemies(player, GameEnum.MovementType.down)
                        enemyMove()  

                if event.key == pygame.K_r:
                    player.rest()
                    map.createEnemiesRest(player)
                    enemyMove()

                if event.key == pygame.K_h:
                    if player.inventory != []:
                        index = None
                        for i in range(len(player.inventory)):
                            if isinstance(player.inventory[i], HelthPotion):
                                index = i
                                
                        if index != None:
                            item = player.inventory.pop(index)
                            player.use(item)
                            
                    else :
                        text =player.name + " doesn't posses health potion."
                        settings.addGameText(text)
                    enemyMove()

                if event.key == pygame.K_m:
                    if isinstance(player, Wizard) and player.inventory != []:
                            index = None
                            
                            for i in range(len(player.inventory)):
                                if isinstance(player.inventory[i],ManaPotion):
                                    index = i
                            if index != None:
                                item = player.inventory.pop(index)
                                player.use(item)
                    if isinstance(player, Warrior):
                        text =player.name + " can't uses mana potion."
                        settings.addGameText(text)
                    if index == None:
                        text =player.name + " doesn't posses mana potion."
                        settings.addGameText(text)
                    enemyMove()
                
                if event.key == pygame.K_SPACE:
                    index = player.enemyToAttack()
                    if index != None:
                        enemy = settings.enemies[index]
                        if player.attack(enemy) and enemy.hitPoints <= 0 and player.experiencePoints <= 13999:
                            settings.tiles[enemy.enemyCurrentPossitionX][enemy.enemyCurrentPossitionY].occupancy = False
                            settings.enemies.pop(index)
                            text =player.name + " kills " + enemy.name + " and earn " + str(enemy.XPreturn) + " XP points"
                            settings.addGameText(text)
                            old_level = player.getLevel()
                            player.addXP(enemy.XPreturn)
                            if old_level < player.getLevel() and player.hitPoints <= (player.maxHitPoints*(2/3)):

                                for i in range(countHealthPotion()//2): 
                                    index = None
                                    for i in range(len(player.inventory)):
                                        if isinstance(player.inventory[i], HelthPotion):
                                            index = i
                                
                                    if index != None:
                                        item = player.inventory.pop(index)
                                        player.use(item)
                            if random.random() <= 0.25:
                                map.addItem(player, enemy.enemyCurrentPossitionX, enemy.enemyCurrentPossitionY)
                    enemyMove()

                if event.key == pygame.K_p:
                    if settings.tiles[player.currentPositionX][player.currentPositionY].store != None:
                        if isinstance(player, Warrior) and isinstance(settings.tiles[player.currentPositionX][player.currentPositionY].store, Sword):
                            sword = settings.tiles[player.currentPositionX][player.currentPositionY].store
                            settings.tiles[player.currentPositionX][player.currentPositionY].store = player.dropWeapon()
                            player.setWeapon(sword)
                        if isinstance(player, Wizard) and isinstance(settings.tiles[player.currentPositionX][player.currentPositionY].store, Staff):
                            staff = settings.tiles[player.currentPositionX][player.currentPositionY].store
                            settings.tiles[player.currentPositionX][player.currentPositionY].store = player.dropWeapon()
                            player.setWeapon(staff)
                        if isinstance(settings.tiles[player.currentPositionX][player.currentPositionY].store, Werdna_Ring):
                            print(player.name + " found Werdna's Ring!! Congratulation")
                            sys.exit()
                    enemyMove()
               
                                
                if settings.tiles[player.currentPositionX][player.currentPositionY].ground == GameEnum.GroundType.stairs:
                    cave = cave + 1
                    makeMap(cave)
                    settings.enemies = []
                    text =player.name + " enters cave No " + str(cave + 1)
                    settings.addGameText(text)
                else:
                    pass

                
                printText()
                printPlayerStatus()
                DrawMap.drawMap()
                DrawMap.drawItem()
                DrawMap.enemyDepiction()
                DrawMap.drawPlayer(player.currentPositionX, player.currentPositionY)
                pygame.display.update()


run_game()
