# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers



class ResidualBlock(Model):

  def __init__(self):
    """Initialize a Residual Block where each layer is of the form F(x) =
    H(x)+x and H(x) is defined as H(x) = relu(x · W1 + b1) · W2 + b2.
    """

    super(ResidualBlock, self).__init__()

    self.ResBlockLayers = [tfkl.Dense(100, activation="relu"), tfkl.Dense(100, activation=None)]
    

  @tf.function 
  def call(self, x):
    """Activate our ResidualBlock, by propagating the input through it layer by layer.

    Arguments:
      x (tensor): Tensor containing the input to our ResBlock

    Returns:
      x (tensor): Tensor containing the activation of our ResBlock
    """

    y = x

    for layer in self.ResBlockLayers:
      y = layer(y)
    
    # Apply the pointwise addition of the original input to the ResBlock's output
    y = x+y

    # Apply the activation function
    return tf.nn.relu(y)
    
    

class Generator(Model):

  def __init__(self):
    """Initialize a Generator that generates fake tweet embeddings.
    """ 

    super(Generator, self).__init__()

    # Since we use a stacked Encoder that returns two embeddings the generator has to do the same.
    # We achieve this by implementing it with two layerlists.
    self.generator_layers_1 = [ResidualBlock() for _ in range(40)]
    self.generator_layers_1.append(tfkl.Dense(600, activation=None))

    self.generator_layers_2 = [ResidualBlock() for _ in range(40)]
    self.generator_layers_2.append(tfkl.Dense(600, activation=None))

  @tf.function
  def call(self, x1, x2): 
    """Activate our Generator propagating the input through it layer by layer.

    Arguments:
      x1 (tensor): Normal distributed noise  
      x2 (tensor): Normal distributed noise 

    Returns:
      x1 (tensor): Recreated the last hidden state of the first encoder lstm
      x2 (tensor): Recreate the last hidden state of the last encoder lstm
    """

    for layer in self.generator_layers_1:
      x1 = layer(x1)

    for layer in self.generator_layers_2:
      x2 = layer(x2) 

    return x1, x2
    


class Discriminator(Model):

  def __init__(self):
    """Initialize a Discriminator that decides whether the input tweet embedding is fake or real 
    """ 

    super(Discriminator, self).__init__()

    # Since we had to adjust the Generator such that it is able to handle two embeddings we also need to adjust the discriminator.
    # We again achieve this by implementing it with two layerlists.
    self.discriminator_layers_1 = [ResidualBlock() for _ in range(40)]
    self.discriminator_layers_1.insert(0, tfkl.Dense(100, activation=None))

    self.discriminator_layers_2 = [ResidualBlock() for _ in range(40)]
    self.discriminator_layers_2.insert(0, tfkl.Dense(100, activation=None))

    self.dense_1 = tfkl.Dense(100, activation=None)
    self.dense_2 = tfkl.Dense(1, activation=None)

  @tf.function
  def call(self, x1, x2): 
    """Activate our Discriminator propagating the input through it layer by layer

    Arguments:
      x1 (tensor): Real or fake tweet embedding 
      x2 (tensor): Real or fake tweet embedding 

    Returns:
      x (tensor): Recreated tweet embedding
    """

    for layer in self.discriminator_layers_1:
      x1 = layer(x1)

    for layer in self.discriminator_layers_2:
      x2 = layer(x2)

    x = self.dense_1(tf.concat((x1, x2), axis=-1))
    x = self.dense_2(x)
    return x