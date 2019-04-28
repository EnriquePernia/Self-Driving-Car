# Self driving car ai w/ tensorflow
# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import random
import os


class NN(tf.keras.Model):

    def __init__(self, input_neurones, actions_output):
        super(NN, self).__init__()
        self.input_neurones = input_neurones
        self.actions_output = actions_output
        x = keras.Input(shape=(4,))
        self.dense1 = keras.layers.Dense(30, activation=tf.nn.relu)(x)
        self.dense2 = keras.layers.Dense(actions_output, activation=tf.nn.softmax)

    def forward(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)  # Q values

x = NN(4,4)
print(x)
print(x.forward([1,4,7,2]))
