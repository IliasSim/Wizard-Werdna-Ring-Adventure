from collections import abc
from typing import NewType
import tensorflow as tf
import numpy as np 
import GamePAI
from tensorflow.keras.layers.experimental import preprocessing


featuresCNN1 = 32
CNN1Shape = 3
CNN1Step =3
denseLayerN = 512
denseLayerNL_2 = 64
denseLayerNL_3= 64
featuresCNN2 = 16
CNN2Shape = 4
CNN2Step = 2
n_step = 4
input = (148,148,3)
class actor(tf.keras.Model):
    '''This class creates the model of the critic part of the reiforcment learning algorithm'''
    def __init__(self):
        super().__init__()
        # self.l1 = tf.keras.layers.Conv2D(32,(8,8),(4,4),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(64,(4,4),(1,1),activation = 'relu',input_shape=(148,148,3))
        # self.l1 = tf.keras.layers.Conv2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step),activation = 'relu',input_shape=input)
        # self.l2 = tf.keras.layers.Conv2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step),activation = 'relu')
        # self.l3 = tf.keras.layers.Flatten()
        # self.l4 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        # self.a = tf.keras.layers.Dense(9, activation ='softmax')
        self.l1 = tf.keras.layers.Conv2D(featuresCNN1,(CNN1Shape,CNN1Shape),(CNN1Step,CNN1Step),activation = 'relu')
        self.l2 = tf.keras.layers.Conv2D(featuresCNN2,(CNN2Shape,CNN2Shape),(CNN2Step,CNN2Step),activation = 'relu')
        self.l3 = tf.keras.layers.Flatten()
        self.l4 = tf.keras.layers.Dense(denseLayerN,activation = 'relu')
        self.l1_2 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        self.l1_21 = tf.keras.layers.Dense(denseLayerNL_2,activation = 'relu')
        self.l1_3 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        self.l1_31 = tf.keras.layers.Dense(denseLayerNL_3,activation = 'relu')
        self.conc = tf.keras.layers.Concatenate(axis=-1)
        self.add = tf.keras.layers.Add()
        self.drop = tf.keras.layers.Dropout(0.2)
        #self.conc = tf.keras.layers.concatenate([self.l4,self.l1_2],axis = -1)
        #self.conc3 = tf.keras.layers.concatenate([self.l1_3,self.conc],axis = -1)
        self.a = tf.keras.layers.Dense(9, activation ='softmax')
      

    def call(self,input_data,input_data1,input_data2):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.l3(x)
        x= self.l4(x)
        y = self.l1_2(input_data2)
        y = self.l1_2(y)
        d = self.l1_3(input_data1)
        d = self.l1_31(d)
        e = self.add((y,d))
        g = self.conc((e,x))
        #e = tf.concat([x,y],axis = -1)
        #f= tf.concat([d,e],axis = -1)
        #e = tf.add(x,y)
        f= self.drop(g)
        a = self.a(f)
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
        x = tf.keras.Input(shape=(148, 148, 3))
        y = tf.keras.Input(shape=(10,))
        z = tf.keras.Input(shape=(125,))
        return tf.keras.Model(inputs=[x,y,z], outputs=self.call(x,y,z))

if __name__ == '__main__':
    act = actor()
    act.model().summary()
    dot_img_file = 'model_conc.png'
    tf.keras.utils.plot_model(act.model(), to_file=dot_img_file, show_shapes=True) 


