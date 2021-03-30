# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers



class ResidualBlock(Model):

  def __init__(self):
    """Initialize a Residual Block where each layer is of the form F(x) =
    H(x)+x and H(x) is defined as H(x) = relu(x Â· W1 + b1) Â· W2 + b2.
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
    self.generator_layers = [ResidualBlock() for _ in range(40)]
    self.generator_layers.append(tfkl.Dense(600, activation=None))


  @tf.function
  def call(self, x): 
    """Activate our Generator propagating the input through it layer by layer.

    Arguments:
      x (tensor): Normal distributed noise  

    Returns:
      x (tensor): Recreated the last hidden state of the encoder lstm
    """

    for layer in self.generator_layers:
      x = layer(x)

    return x
    


class Discriminator(Model):

  def __init__(self):
    """Initialize a Discriminator that decides whether the input tweet embedding is fake or real 
    """ 

    super(Discriminator, self).__init__()

    # Since we had to adjust the Generator such that it is able to handle two embeddings we also need to adjust the discriminator.
    # We again achieve this by implementing it with two layerlists.
    self.discriminator_layers = [ResidualBlock() for _ in range(40)]
    self.discriminator_layers.insert(0, tfkl.Dense(100, activation=None))
    self.discriminator_layers.append(tfkl.Dense(1, activation=None))


  @tf.function
  def call(self, x): 
    """Activate our Discriminator propagating the input through it layer by layer

    Arguments:
      x (tensor): Real or fake tweet embedding 

    Returns:
      x (tensor): Recreated tweet embedding
    """

    for layer in self.discriminator_layers:
      x = layer(x)

    return x