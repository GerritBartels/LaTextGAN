# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
plt.style.use('ggplot') # change the style of the plots to a nicer theme
import random
import time
# From IPython.display we import clear_output() in order to be able to clear the print statements after each epoch
from IPython.display import clear_output
from tqdm import tqdm, tqdm_notebook # show progress bar
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
  
  
  
def visualization(word2vec_model, train_losses, test_losses, input_tweet, predicted_tweet, num_epochs): 
  """Visualize performance and loss for training and test data. 
  
  Arguments:
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    train_losses (list): List containing the training losses of the Network
    test_losses (list): List containing the losses of the Network over the test data
    input_tweet (tuple): Tuple containing the to be Autoencoded tweets 
    predicted_tweet (tuple): Tuple containing the reconstructed tweets for visualizing the progress of the Network 
    num_epochs (int): Current Epoch
  """ 

  # We use the inbuilt index2word from the word2vec model to convert the lists of indices back into their respective tokens
  print("Autoencoded Tweet (Training Sample):")
  # Minus 1 since we artificially inserted a 0 column into the embedding matrix (for padding)
  print(f"Input: {' '.join([word2vec_model.wv.index2word[i-1] for i in input_tweet[0] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index2word[i-1] for i in tf.argmax(predicted_tweet[0], axis=2).numpy()[0] if i != 0])}")
  print()
  print("Autoencoded Tweet (Training Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index2word[i-1] for i in input_tweet[1] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index2word[i-1] for i in tf.argmax(predicted_tweet[1], axis=2).numpy()[0] if i != 0])}")
  print()
  
  print("Autoencoded Tweet (Test Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index2word[i-1] for i in input_tweet[2] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index2word[i-1] for i in tf.argmax(predicted_tweet[2], axis=2).numpy()[0] if i != 0])}")
  print()
  print("Autoencoded Tweet (Test Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index2word[i-1] for i in input_tweet[3] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index2word[i-1] for i in tf.argmax(predicted_tweet[3], axis=2).numpy()[0] if i != 0])}")
  print()
  print()


  # Plot for visualizing the average loss over the training and test data
  fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
  ax1.plot(train_losses, label='training')
  ax1.plot(test_losses, label='test')
  ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
  ax1.legend()
  plt.show()



@tf.function(experimental_relax_shapes=True)
def train_step_ae(model, input, target, teacher, loss_function, optimizer):
  """Perform a training step for a given Network by
  1. Propagating the input through the network
  2. Calculating the loss between the networks output and the true targets
  3. Performing Backpropagation and Updating the trainable variables witht the calculated gradients 
 
  Arguments:
    model (AutoEncoder): given instance of an initialised  Network with all its parameters
    input (tensor): Tensor containing the input data for the encoder
    target (tensor): Tensor containing the respective targets 
    teacher (tensor): Tensor containing the input data for the decoder
    loss_function (keras.losses): function from keras to calculate the loss
    optimizer (keras.optimizers): function from keras defining the to be applied optimizer during learning 
 
  Returns:
    loss (tensor): Tensor containing the loss of the Network 
  """

  with tf.GradientTape() as tape:
    # 1.
    prediction = model(input, teacher)
    # 2.
    loss = loss_function(target, prediction)
    # 3.
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
 
  return loss



def test_ae(model, test_data, loss_function):
  """Tests the models loss over the given data set with a given loss_function
 
  Arguments:
    model (Model): given instance of an initialised Network with all its parameters
    test_data (Dataset): test dataset to test the NN on 
    loss_function (keras.losses): function from keras to calculate the loss 
 
  Returns:
    test_loss (float): Average loss of the Network over the test set
  """

  test_loss_aggregator = []
  
  for input, target, teacher in test_data:
    prediction = model(input, teacher)
    sample_test_loss = loss_function(target, prediction)
    test_loss_aggregator.append(sample_test_loss)
 
  test_loss = tf.reduce_mean(test_loss_aggregator)
  
  return test_loss



def trainModel(model, word2vec_model: gensim.models.word2vec.Word2Vec, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, loss_function: tf.keras.losses, num_epochs: int=50, learning_rate: float=0.0005, running_average_factor: float=0.95): 
  """Function that implements the training algorithm for a given Network.
  Prints out useful information and visualizations per epoch.

  Arguments:
    network (AutoEncoder): Model that the training algorithm should be applied to
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    train_dataset (tf.data.Dataset): Dataset to perform training on
    test_dataset (tf.data.Dataset): Dataset to perform testing on
    loss_function (keras.losses): To be applied loss_function during training
    num_epochs (int): Defines the amount of epochs the training is performed
    learning_rate (float): To be used learning rate, per default set to 0.001
    running_average (float): To be used factor for computing the running average of the trainings loss, per default set to 0.95
  """ 

  tf.keras.backend.clear_session()

  # Extract two fixed tweets from the training and test dataset each for visualization during training.
  for input, target, teacher in train_dataset.take(1):
  	train_tweet_for_visualisation_1 = (input[0], teacher[0])
  	train_tweet_for_visualisation_2 = (input[1], teacher[1])
    
  for input, target, teacher in test_dataset.take(1):
  	test_tweet_for_visualisation_1 = (input[0], teacher[0])
  	test_tweet_for_visualisation_2 = (input[1], teacher[1])
    
  # Initialize the optimizer: Adam with custom learning rate.
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  
  # Initialize lists for later visualization.
  train_losses = []
  test_losses = []
  
  # testing once before we begin on the test and train data
  test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function)
  test_losses.append(test_loss)

  train_loss = test_ae(model=model, test_data=train_dataset, loss_function=loss_function)
  train_losses.append(train_loss)


  for epoch in range(num_epochs):
    start = time.time()

    # Training and computing running average
    running_average = 0
    pbar = tqdm(total=519)
    for input, target, teacher in train_dataset:
        train_loss = train_step_ae(model=model, input=input, target=target, teacher=teacher, loss_function=loss_function, optimizer=optimizer)
        running_average = running_average_factor * running_average  + (1 - running_average_factor) * train_loss
        pbar.update(1)
    pbar.close()
    
    train_losses.append(running_average)

    # Testing
    test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function)
    test_losses.append(test_loss)
    
    # Print useful information
    clear_output()
    print(f"Epoch: {str(epoch+1)}")      
    print()
    print(f"This epoch took {timing(start)} seconds")
    print()
    print(f"Training loss for current epoch: {train_losses[-1]}")
    print()
    print(f"Test loss for current epoch: {test_losses[-1]}")
    print()

    # Feed sample tweets through the network for visualization
    train_pred_tweet_1 = model(tf.expand_dims(train_tweet_for_visualisation_1[0], axis=0), tf.expand_dims(train_tweet_for_visualisation_1[1], axis=0))
    train_pred_tweet_2 = model(tf.expand_dims(train_tweet_for_visualisation_2[0], axis=0), tf.expand_dims(train_tweet_for_visualisation_2[1], axis=0))
    test_pred_tweet_1 = model(tf.expand_dims(test_tweet_for_visualisation_1[0], axis=0), tf.expand_dims(test_tweet_for_visualisation_1[1], axis=0), training=False)
    test_pred_tweet_2 = model(tf.expand_dims(test_tweet_for_visualisation_2[0], axis=0), tf.expand_dims(test_tweet_for_visualisation_2[1], axis=0), training=False)

    visualization(word2vec_model,
                  train_losses=train_losses, 
                  test_losses=test_losses, 
                  input_tweet=(train_tweet_for_visualisation_1[0],train_tweet_for_visualisation_2[0], test_tweet_for_visualisation_1[0], test_tweet_for_visualisation_2[0]), 
                  predicted_tweet=(train_pred_tweet_1, train_pred_tweet_2, test_pred_tweet_1, test_pred_tweet_2), 
                  num_epochs=epoch+1)


  print()
  model.summary()

