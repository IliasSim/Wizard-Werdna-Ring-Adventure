# **Wizard Werdna Ring**
The code is written in python and creates a rogulike game.

  - [**Scope**](#scope)
  - [**Caves/Enviroment**](#cavesenviroment)
  - [**Player**](#player)
    - [**Player Depiction**](#player-depiction)
    - [**Player Visiblity**](#player-visiblity)
    - [**Player Actions**](#player-actions)
      - [**Movement of the Player**](#movement-of-the-player)
      - [**Attack**](#attack)
      - [**Rest**](#rest)
      - [**Health potion usage**](#health-potion-usage)
      - [**Mana potion usage**](#mana-potion-usage)
      - [**Picking up weapons**](#picking-up-weapons)
    - [](#)
## **Scope**
Main goal of the game is the player to find the Wizard Werdna Ring. In order to achive this the player searche a cave complex. Each cave is inabited by hostile creatures ready to attack our player. The ring is located at the tenth and final cave.

## **Caves/Enviroment**
The enviroment of the game consist of caves, each cave is a 2d space of tiles. The tiles can be floor (where the players and the enemies can walk), wall and stairs. The stair tile is the entrance for the next cave. The color for each tile is red for the floor, black for the wall and yellow for the stair. 

![Wizard_Werdna_Ring_Map](images/Wizard_Werdna_Ring_Map.png)
*An example of cave depiction*

## **Player**
You can choose between two types of player at the first screen of the game by pressing 1 for the Warrior type and 2 for the Wizard. A screenshot of the introductory screen is given below.

![Wizard_Werdna_Ring](images/Wizard_Werdna_Ring.png)

In the second screen you can insert the name of the player. Screenshot is given below.
![Wizard_Werdna_Ring_Name](images/Wizard_Werdna_Ring_Name.png)

### **Player Depiction**
The player is depicted as a blue rectangle.

![Wizard_Werdna_Ring_depiction](images/Wizard_Werdna_Ring_depiction.png)

### **Player Visiblity**
The player can see for a diastance equal with 6 map tiles and that means that the player can see if the tiles are floor, stair or wall and also if there are any enemies or items. For the tiles that are outside the visibility range there are two visibilty categories, the unknown which means that the player can't see anything for this tile and the fogged which means a tile that the player can see only if it is a wall or floor and if the tile has any item (the enemies are invisible in this case).

![Wizard_Werdna_Ring_visibility](images/Wizard_Werdna_Ring_visibility.png)
*A depiction of the game after some movements of the player is given above. The bright red for the tiles means that are inside the player's visibilty and the les opaque red means that the tiles visibility type is fogged.*

### **Player Actions**
The game action follows the player actions, in simple words first the player performs one action and in response the environment reacts.
#### **Movement of the Player**
Generally the player can move in the four directions using the keys w to move up,a to move left,s to move down and d to move right. For each movement the player moves for one tile in the map/cave.

#### **Attack**
In order to attack you have to hit the space button.
Each type of player perform different type of attack. The Warrior use sword and performs attack on an enemy only if he is on the next tile, if there are more tha one enemy nearby the player attacks the one with the less hitpoints. The wizard use staff as weapon and use spel to attack. For this reason the wizard can attack any enemy inside his visibility and if there are more than one enemy the attack priority is distance and then the enemy hitpoints. Each time the wizard attacks he consum 5 mana points.
The damage that the player performs depends on the player level and the weapon the player posses. The attribute that characterize the damage the player can perform is the strength for the warrior and the inteligence for the wizard.

#### **Rest**
By pressing the r key the player rest and earns 4 hitpoints. If he is a wizard earns and 4 mana points. Keep in mind that each time you are pressing the r button there is a 25% possibility an enemy to appear.  

#### **Health potion usage**
You have to press h in order to use one health potion and earn 20 hitpoints.

#### **Mana potion usage**

This action is performed only when the player is wizard type. By pressing the m key the player uses one mana potion and earns 20 mana points.

#### **Picking up weapons**
When the palyer moves in a tile that has a weapon the player can pick that weapon by pressing the p key. The sword weapon that the warrion can aquire add two boost one to the maximum hitpoints the player can has and one to the player strenght. The staff is the weapon that the wizard can aquire and add three boost one to the maximum hitpoints, to the maximum mana points and to the inteligence. 

### **Player level**
The player has level starting from the first and ending at the fifth. Each level determines the player max hitpoints for both types of player, max mana points and inteligence for the wizard type of player and strength for the warrior type. Also the level determines the enemies the player encounter and the weapons he can find. In order to change level the player gathers experience points by killing enemies. In the following tables are presented the level of the player and the corresponding required experience points, in order the player to reach each level and the attributes the player has at each level.

|Level  |Experience Points  |Hitpoints  |Strength  |
|:-----:|:-----------------:|:---------:|:--------:|
|   1   |  0-299            |     30    |    10    |
|   2   |  300-899          |     60    |    20    |
|   3   |  900-2699         |     80    |    25    |
|   4   |  2700 - 6499      |     90    |    30    |
|   5   |  6500 - 1399      |     100   |    35    |

Table 1: Warrior level and corresponding attributes

|Level  |Experience Points  |Hitpoints  |Mana points| Inteligence|
|:-----:|:-----------------:|:---------:|:--------:|:-----------:|
|   1   |  0-299            |     20    |    30    |      10     |
|   2   |  300-899          |     40    |    50    |      20     |
|   3   |  900-2699         |     50    |    70    |      30     |
|   4   |  2700 - 6499      |     55    |    90    |      40     |
|   5   |  6500 - 1399      |     60    |    110   |      50     |

Table 2: Wizard level and corresponding attributes

## **Enemies**