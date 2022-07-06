import pygame
import random
import matplotlib.pyplot as plt
import GameSettings as settings
import pygame
import numpy as np
import torch
from pytorch_qn_human_repaly import Sequence_Buffer
from os.path import exists
import pickle
import os
buffer = Sequence_Buffer()
process_cont = True
agent = 0
g = 0
if exists('D:\ekpa\diplomatiki\ilias.txt'):
    size = os.path.getsize('D:\ekpa\diplomatiki\ilias.txt') 
else: 
    with open('D:\ekpa\diplomatiki\ilias.txt', 'x') as f:
        f.write('')
        size = 0
if size >0:
    with open ('D:\ekpa\diplomatiki\ilias.txt', 'rb') as fp:
        buffer.buffer = pickle.load(fp)
state = np.reshape(buffer.buffer[0][3],(148,148,3)) 
state32 = state.astype('float32')
fig = plt.figure()
plt.imshow(state32)
plt.title("Plot 2D array")
plt.show()  


 
    
