truncate = 32
memory_size = 350000
from os.path import exists
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



class Sequence_Buffer:
    def __init__(self):
        self.buffer_size = truncate
        self.buffer = []
        self.epsilon = 1

    def eraseMemory(self):
        self.buffer = []

    def WriteMemory(self,last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action):
        if len(self.buffer) >= memory_size:
            del self.buffer[0]
        self.buffer.append([last_state,last_playerStatus,last_gameText,next_state, next_playerStatus, next_gameText,reward,action])

buffer = Sequence_Buffer()
if exists('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt'):
        size = os.path.getsize('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt') 
else: 
    with open('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'x') as f:
        f.write('')
        size = 0
if size >0:
    with open ('D:\ekpa\diplomatiki\date_30_6_22_hmemory\h_memory.txt', 'rb') as fp:
        buffer.buffer = pickle.load(fp)
    
print(buffer.buffer[-1][3].shape,buffer.buffer[-1][6])
state = np.reshape(buffer.buffer[-1][3],(148,148,3)) 
state32 = state.astype('float32')
fig = plt.figure()
plt.imshow(state32)
plt.title("Plot 2D array")
plt.show()  
state = np.reshape(buffer.buffer[-1][0],(148,148,3)) 
state32 = state.astype('float32')
fig = plt.figure()
plt.imshow(state32)
plt.title("Plot 2D array")
plt.show() 
   
