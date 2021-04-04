# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
plt.style.use('ggplot') # Change the style of the plots to a nicer theme
import random
import time
# From IPython.display we import clear_output() in order to be able to clear the print statements after each epoch
from IPython.display import clear_output
from tqdm import tqdm, tqdm_notebook # Show progress bar
import gensim



def timing(start):
  """Function to time the duration of each epoch

  Arguments:
    start (time): Start time needed for computation 
  
  Returns:
    time_per_training_step (time): Rounded time in seconds 
  """
  now = time.time()
  time_per_training_step = now - start
  return round(time_per_training_step, 4)



def discriminator_loss(real_tweet, fake_tweet):
  """Calculate the Wasserstein loss for the discriminator but swapping the sign in order to apply gradient descent.

  Arguments:
    real_tweet (tensor): Linear output from discriminator
    fake_tweet (tensor): Linear output from discriminator

  Returns:
    x (tensor): Wasserstein Loss
  """

  loss_real = - tf.reduce_mean(real_tweet)
  loss_fake = tf.reduce_mean(fake_tweet)

  return loss_real + loss_fake



def generator_loss(fake_tweet):
  """Calculate the Wasserstein loss for the generator.

  Arguments:
    fake_tweet (tensor): Linear output from discriminator

  Returns:
    x (tensor): Wasserstein Loss
  """

  loss_fake = - tf.reduce_mean(fake_tweet)

  return loss_fake
  
  
  
@tf.function() 
def gradient_penalty(discriminator, real_tweet, generated_tweet):
  """Visualize performance of the Generator by feeding predefined random noise vectors through it.
  
  Arguments:
    discriminator (Discriminator): Discriminator class instance
    real_tweet (tensor): Real tweet embedding from Encoder
    generated_tweet (tensor): Fake tweet embedding from Generator

  Return: 
    penalty (): Gradient penalty that will be added to discriminator loss
  """ 

  # Due to the stacked approach we chose for the Autoencoder we had to alter the gradient
  # penalty by interpolating twice and calculating an average penalty. 
  alpha = tf.random.uniform(shape=[real_tweet.shape[0], 1], minval=0, maxval=1)

  interpolate = alpha*real_tweet + (1-alpha)*generated_tweet

  output = discriminator(interpolate)

  gradients = tf.gradients(output, interpolate)

  gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients)))

  penalty = 10*tf.reduce_mean((gradient_norm-1.)**2)

  return penalty



def visualize_GAN(autoencoder, word2vec_model, fixed_input, random_input, train_losses_generator, train_losses_discriminator, num_epochs):
  """Visualize performance of the Generator by feeding predefined random noise vectors through it.
  
  Arguments:
    autoencoder (AutoEncoder): AutoEncoder class instance
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    fixed_input (tensor): List containing predefined random vectors
    random_input (tensor): List containing predefined random vectors
    train_losses_generator (list): List containing the generator losses
    train_losses_discriminator (list): List containing the discriminator losses 
    num_epochs (int): Current Epoch
  """ 

  print()
  print(f"From Fixed Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=fixed_input[0], training=False) if i.numpy()[0] != 0])}")
  print(f"From Fixed Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=fixed_input[1], training=False) if i.numpy()[0] != 0])}")
  print()
  print(f"From Random Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=random_input[0], training=False) if i.numpy()[0] != 0])}")
  print(f"From Random Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=random_input[1], training=False) if i.numpy()[0] != 0])}")

  plt.style.use('ggplot')
  
  fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
  ax1.plot(train_losses_generator, label='Generator')
  ax1.plot(train_losses_discriminator, label='Discriminator')
  ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
  if num_epochs>25 and num_epochs<=50:
      ax1.set_ylim([-10,100])
  if num_epochs>50:
      ax1.set_ylim([-5,25])
  ax1.legend()
  
  plt.show()



@tf.function()  
def train_step_GAN(generator, discriminator, train_data, optimizer_generator, optimizer_discriminator, train_generator):
  """Perform a training step for a given GAN Network by
  1. Generating random noise for the Generator
  2. Feeding the noise through the Generator to create fake tweet embeddings for the Discriminator 
  3. Feeding the fake and real tweet embeddings through the Discriminator 
  4. Calculating the loss for the Disriminator and the Generator 
  5. Performing Backpropagation and Updating the trainable variables with the calculated gradients, using the specified optimizers

  Arguments:
    generator (Generator): Generator class instance
    discriminator (Discriminator): Discriminator class instance
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    train_data (tf.data.Dataset): Real tweet embedding from Encoder
    optimizer_generator (tf.keras.optimizers): function from keras defining the to be applied optimizer during training
    optimizer_discriminator (tf.keras.optimizers): function from keras defining the to be applied optimizer during training
    train_generator (bool): Whether to update the generator or not
 
  Returns:
    loss_from_generator, loss_from_discriminator (Tupel): Tupel containing the loss of both the Generator and Discriminator
  """

  # 1.
  noise = tf.random.normal([train_data.shape[0], 100])

  # Two Gradient Tapes, one for the Discriminator and one for the Generator 
  with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
      # 2.
      generated_tweet = generator(noise)

      # 3.
      real = discriminator(train_data)
      fake = discriminator(generated_tweet)

      # 4.
      loss_from_generator = generator_loss(fake)
      # Add gradient penalty to enforce lipschitz continuity
      loss_from_discriminator = discriminator_loss(real, fake) + gradient_penalty(discriminator=discriminator, real_tweet=train_data, generated_tweet=generated_tweet)

  # 5.
  gradients_from_discriminator = discriminator_tape.gradient(loss_from_discriminator, discriminator.trainable_variables)
  optimizer_discriminator.apply_gradients(zip(gradients_from_discriminator, discriminator.trainable_variables))

  # We update the generator once for ten updates to the discriminator
  if train_generator:
    gradients_from_generator = generator_tape.gradient(loss_from_generator, generator.trainable_variables)
    optimizer_generator.apply_gradients(zip(gradients_from_generator, generator.trainable_variables))

  return loss_from_generator, loss_from_discriminator
  


def train_GAN(generator, discriminator, autoencoder, word2vec_model: gensim.models.word2vec.Word2Vec, train_dataset_GAN: tf.data.Dataset, num_epochs: int=150, running_average_factor: float=0.95, learning_rate: float=0.0001):
  """Function that implements the training algorithm for a GAN.

  Arguments:
    generator (Generator): Generator class instance
    discriminator (Discriminator): Discriminator class instance
    autoencoder (AutoEncoder): AutoEncoder class instance
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    train_dataset_GAN (tf.data.Dataset): Dataset to perform training on
    num_epochs (int): Defines the amount of epochs the training is performed
    learning_rate (float): To be used learning rate, per default set to 0.001
    running_average (float): To be used factor for computing the running average of the trainings loss, per default set to 0.95
  """ 

  tf.keras.backend.clear_session()

  # Two optimizers one for the generator and of for the discriminator
  optimizer_generator=tf.keras.optimizers.Adam(learning_rate=learning_rate)
  optimizer_discriminator=tf.keras.optimizers.Adam(learning_rate=learning_rate)

  # Fixed, random vectors for visualization
  fixed_generator_input_1 = tf.random.normal([1, 100])
  fixed_generator_input_2 = tf.random.normal([1, 100])

  # Initialize lists for later visualization.
  train_losses_generator = []
  train_losses_discriminator = []

  train_generator = False

  for epoch in range(num_epochs):

    start = time.time()
    running_average_gen = 0
    running_average_disc = 0

    with tqdm(total=519) as pbar:
      for batch_no, input in enumerate(train_dataset_GAN):

        # Boolean used to train the discriminator 10x more often than the generator
        train_generator = False
        if batch_no % 10 == 0:
          train_generator = True

        gen_loss, disc_loss = train_step_GAN(generator, discriminator, train_data=input, optimizer_generator=optimizer_generator, optimizer_discriminator=optimizer_discriminator, train_generator=train_generator)
        running_average_gen = running_average_factor * running_average_gen + (1 - running_average_factor) * gen_loss
        running_average_disc = running_average_factor * running_average_disc + (1 - running_average_factor) * disc_loss
        pbar.update(1)

    train_losses_generator.append(float(running_average_gen))
    train_losses_discriminator.append(float(running_average_disc))

    clear_output()
    print(f'Epoch: {epoch+1}')      
    print()
    print(f'This epoch took {timing(start)} seconds')
    print()
    print(f'The current generator loss: {round(train_losses_generator[-1], 4)}')
    print()
    print(f'The current discriminator loss: {round(train_losses_discriminator[-1], 4)}')
    print()

    # Random vectors for visualization that are sampled each epoch
    random_generator_input_1 = tf.random.normal([1, 100])
    random_generator_input_2 = tf.random.normal([1, 100])
    
    visualize_GAN(autoencoder=autoencoder,
                  word2vec_model=word2vec_model,
                  fixed_input=(generator(fixed_generator_input_1), generator(fixed_generator_input_2)), 
                  random_input=(generator(random_generator_input_1), generator(random_generator_input_2)), 
                  train_losses_generator=train_losses_generator, 
                  train_losses_discriminator=train_losses_discriminator, 
                  num_epochs=epoch+1)