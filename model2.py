from collections import abc
from typing import NewType
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers.experimental import preprocessing


featuresCNN1 = 16
CNN1Shape = 4
CNN1Step = 4
featuresCNN2 = 32
CNN2Shape = 2
CNN2Step = 2
featuresCNN3 = 64
CNN3Shape = 5
CNN3Step = 2
denseLayerN = 512
denseLayerNL_2 = 32
denseLayerNL_3= 64
denseLayerNL_21 = 128
denseLayerNL_31 = 128
dropoutRate1 = 0.3
dropoutRate2 = 0.3
n_step = 4
input = (148,148,3)
class actor(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        # self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(64,(4,4),(1,1),activation = 'relu',input_shape=(148,148,3))
        #self.l1 = tf.keras.layers.ConvLSTM2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step),return_sequences = True,data_format='channels_last')
        #self.l2 = tf.keras.layers.ConvLSTM2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step),return_sequences = True,data_format='channels_last')
        # self.l3 = tf.keras.layers.Flatten()
        # self.l4 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        # self.a = tf.keras.layers.Dense(9, activation ='softmax')
        self.l1 = tf.keras.layers.Conv3D(featuresCNN1,(1,CNN1Shape,CNN1Shape),(1,CNN1Step,CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv3D(featuresCNN2,(1,CNN2Shape,CNN2Shape),(1,CNN2Step,CNN2Step),activation = 'relu')
        #self.l3 = tf.keras.layers.Conv3D(featuresCNN3,(1,CNN3Shape,CNN3Shape),(1,CNN3Step,CNN3Step),activation = 'relu')
        self.l4 = tf.keras.layers.Reshape((8,10368))
        self.l5 = tf.keras.layers.LSTM(denseLayerN)
        self.l6 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        self.l7 = tf.keras.layers.LSTM(denseLayerNL_21)
        self.l8 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        self.l9 = tf.keras.layers.LSTM(denseLayerNL_31)
        self.conc1 = tf.keras.layers.Concatenate(axis=-1)
        self.conc2 = tf.keras.layers.Concatenate(axis=-1)
        #self.drop1 = tf.keras.layers.Dropout(dropoutRate1)
        #self.drop2 = tf.keras.layers.Dropout(dropoutRate2)
        #self.conc = tf.keras.layers.concatenate([self.l4,self.l1_2],axis = -1)
        #self.conc3 = tf.keras.layers.concatenate([self.l1_3,self.conc],axis = -1)
        self.a = tf.keras.layers.Dense(9, activation ='softmax')
      

    def call(self,input_data,input_data1,input_data2):
        x = self.l1(input_data)
        x = self.l2(x)
        #x= self.l3(x)
        x = self.l4(x)
        x= self.l5(x)
        y = self.l6(input_data1)
        y = self.l7(y)
        #y = self.drop1(y)
        z = self.l8(input_data2)
        z = self.l9(z)
        #z = self.drop2(z)
        h = self.conc1((y,z))
        #e = self.add((y,d))
        g = self.conc2((h,x))
        a = self.a(g)
        return a

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
    def model(self):
        x = tf.keras.Input(shape=(8,148, 148, 3))
        y = tf.keras.Input(shape=(8,10))
        z = tf.keras.Input(shape=(8,125))
        return tf.keras.Model(inputs=[x,y,z], outputs=self.call(x,y,z))

if __name__ == '__main__':
    act = actor()
    act.model().summary()
    dot_img_file = 'model_conc.png'
    tf.keras.utils.plot_model(act.model(), to_file=dot_img_file, show_shapes=True) 


