# Wizard Werdna Ring
The code is written in python and creates a rogulike game. 
## Scope
Main goal of the game is the player to find the Wizard Werdna Ring. In order to achive this the player searche a cave complex. Each cave is inabited by hostile creatures ready to attack our player. The ring is located at the tenth and final cave.

## Player
You can choose between two types of player at the first screen of the game by pressing 1 for the Warrior type and 2 for the Wizard. A screenshot of the introductory screen is given below.

![Wizard_Werdna_Ring](images/Wizard_Werdna_Ring.png)

In the second screen you can insert the name of the player. Screenshot is given below.
![Wizard_Werdna_Ring_Name](images/Wizard_Werdna_Ring_Name.png)

## Player Depiction
The player is depicted as a blue rectangle.

![Wizard_Werdna_Ring_depiction](images/Wizard_Werdna_Ring_depiction.png)

## Player Visiblity
The player can see for a diastance equal with 6 map tiles and that means that the player can see if the tiles are floor, stair or wall and also if there are any enemies or items. For the tiles that are outside the visibility range there are two visibilty categories, the unknown which means that the player can't see anything for this tile and the fogged which means a tile that the player can see only if it is a wall or floor and if the tile has any item (the enemies are invisible in this case).

![Wizard_Werdna_Ring_visibility](images/Wizard_Werdna_Ring_visibility.png)
*A depiction of the game after some movements of the player is given above. The bright red for the tiles means that are inside the player's visibilty and the les opaque red means that the tiles visibility type is fogged.*

## Player Actions
The game action follows the player actions, in simple words first the player performs one action and in response the environment reacts.
### Movement of the Player
Generally the player can move in the four directions using the keys w to move up,a to move left,s to move down and d to move right. For each movement the player moves for one tile in the map/cave.

### Attack
In order to attack you have to hit the space button.
Each type of player perform different type of attack. The Warrior use sword and performs attack on an enemy  