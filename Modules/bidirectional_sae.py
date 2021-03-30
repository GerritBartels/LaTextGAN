# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers



class Encoder(Model):
  # We decided to use type hints whenever the user has the possibility to provide input
  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, bidirectional: bool=True, embedding_size: int=200):
    """Initialize the Encoder that creates an embedding of tweets

    Arguments:
      vocab_size (int): Defines the input dimensionality of the embedding layer
      embedding_matrix (ndarray): Sets the weights of the embedding layer
      bidirectional (bool): Whether to use bidirectional LSTMs or not
      embedding_size (int): Defines the output dimensionality of the embedding layer
    """ 

    super(Encoder, self).__init__()

    # We decided to pretrain our embeddings and therefore set the "trainable" parameter to False due to the limited amount of data at hand.
    # The mask_zero argument is set to True such that the layer can work with padded batches
    self.bidirectional = bidirectional
    self.embedding = tfkl.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, weights=[embedding_matrix], trainable=False, mask_zero=True)
    if bidirectional:
        self.bi_lstm_1 = tfkl.Bidirectional(tfkl.LSTM(units=100, return_sequences=True, return_state=True))
        self.bi_lstm_2 = tfkl.Bidirectional(tfkl.LSTM(units=100))
    else:
        self.lstm_1 = tfkl.LSTM(units=100, return_sequences=True, return_state=True)
        self.lstm_2 = tfkl.LSTM(units=100)
    self.dense_1 = tfkl.Dense(units=600, activation=None)
    self.dense_2 = tfkl.Dense(units=600, activation=None)



  @tf.function(experimental_relax_shapes=True)
  def call(self, x, training: bool=True):
    """Activate our Encoder propagating the input through it layer by layer

    Arguments:
      x (tensor): Tensor containing the input to our Encoder
      training (bool): Indicates whether regularization methods should be used or not when calling the Encoder 

    Returns:
      dense_out_1 (tensor): Tensor containing the last hidden state of the first lstm projected to the state size of the decoder
      dense_out_2 (tensor): Tensor containing the last hidden state of the last lstm projected to the state size of the decoder
    """

    x = self.embedding(x)
    # Since the bidirectional encoder returns two elements more than the standard stacked variant,
    # we need to differentiate their call.
    if self.bidirectional:
        hidden_states, forward_hs_1, _, backward_hs_1, _ = self.bi_lstm_1(x, training=training)
        hs_1 = tf.concat([forward_hs_1, backward_hs_1], axis=-1)
        hs_2 = self.bi_lstm_2(hidden_states, training=training)
    else:
        hidden_states, hs_1, _ = self.lstm_1(x, training=training)
        hs_2 = self.lstm_2(hidden_states, training=training)
    # Use this dense layer to project the hidden state size of the last encoding step (100) to the state size of the decoder (600) 
    dense_out_1 = self.dense_1(hs_1, training=training) 
    dense_out_2 = self.dense_2(hs_2, training=training)

    return dense_out_1, dense_out_2



class Decoder(Model):
  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=200):
    """Initialize the Decoder that recreates tweets based on the embeddings of the Encoder

    Arguments:
      vocab_size (int): Defines the input dimensionality of the embedding layer
      embedding_matrix (ndarray): Sets the weights of the embedding layer
      embedding_size (int): Defines the output dimensionality of the embedding layer
    """  

    super(Decoder, self).__init__()

    self.embedding = tfkl.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, weights=[embedding_matrix], trainable=False, mask_zero=True)
    self.lstm_1 = tfkl.LSTM(units=600, return_sequences=True, return_state=True)
    self.lstm_2 = tfkl.LSTM(units=600, return_sequences=True, return_state=True)
    # Use the dense layer to project back onto vocab size and apply softmax in order to obtain a probability distribution over all tokens
    self.dense = tfkl.Dense(units=vocab_size+1, activation="softmax")



  @tf.function(experimental_relax_shapes=True)  
  def call(self, x, states_1, states_2, training: bool=True):
    """Activate our Decoder propagating the input through it by using teacher forcing

    Arguments:
      x (tensor): Tensor containing the teacher input
      states_1(tensor): Tensor containing the last hidden state of the first lstm from the Encoder 
      states_2(tensor): Tensor containing the last hidden state of the last lstm from the Encoder 
      training (bool): Indicates whether regularization methods should be used or not when calling the Decoder 

    Returns:
      dense_out (tensor): Tensor containing the reconstructed input
    """

    x = self.embedding(x)
    hidden_states_1, _, _ = self.lstm_1(x, initial_state=[states_1, tf.zeros_like(states_1)], training=training)
    hidden_states_2, _, _ = self.lstm_2(hidden_states_1, initial_state=[states_2, tf.zeros_like(states_2)], training=training)
    dense_out = self.dense(hidden_states_2, training=training)
    return dense_out



  def inference_mode(self, states_1, states_2, training: bool=True):
    """Call Decoder in inference mode: Reconstructing the input using only start token and embeddings. 
    Each Decoder step gets the previous prediction of the Decoder as additional input.

    Arguments:
      states_1(tensor): Tensor containing the last hidden state of the first lstm from the Encoder 
      states_2(tensor): Tensor containing the last hidden state of the last lstm from the Encoder 
      training (bool): Indicates whether regularization methods should be used or not when calling the Encoder 

    Returns:
      predictions (List): List containing the reconstructed input
    """

    predictions = []
    start_token = self.embedding(tf.constant([[2]]))
    _, hs_1, cs_1 = self.lstm_1(start_token, initial_state=[states_1, tf.zeros_like(states_1)], training=training)
    _, hs_2, cs_2 = self.lstm_2(tf.expand_dims(input=hs_1, axis=0), initial_state=[states_2, tf.zeros_like(states_2)], training=training)
    dense_out = self.dense(hs_2, training=training)
    pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1)
    predictions.append(pred)

    max_seq_length = 78 
    end_token = 3
    stopping_criterion = False

    # The stopping criterion is used to tell the Decoder when to stop predicting new tokens.
    # (either when eos token has been generated or the max sequence length has been reached)
    while not stopping_criterion:

      last_pred = self.embedding(tf.expand_dims(pred, axis=0))
      _, hs_1, cs_1 = self.lstm_1(last_pred, initial_state=[hs_1, cs_1], training=training)
      _, hs_2, cs_2 = self.lstm_2(tf.expand_dims(input=hs_1, axis=0), initial_state=[hs_2, cs_2], training=training)
      dense_out = self.dense(hs_2, training=training)
      pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1) 
      predictions.append(pred)

      if pred  == end_token or len(predictions) >= max_seq_length:
        stopping_criterion=True

    return predictions



class AutoEncoder(Model):

  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, bidirectional: bool=True, embedding_size: int=200):
    """Initialize an Autoencoder consisting of an Encoder and Decoder

    Arguments:
      vocab_size (int): Defines the input dimensionality of the embedding layer
      embedding_matrix (ndarray): Sets the weights of the embedding layer
      bidirectional (bool): Whether to use bidirectional LSTMs or not
      embedding_size (int): Defines the output dimensionality of the embedding layer
    """  

    super(AutoEncoder, self).__init__()

    self.Encoder = Encoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, bidirectional=bidirectional, embedding_size=embedding_size)
    self.Decoder = Decoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size)


  @tf.function(experimental_relax_shapes=True)    
  def call(self, input, teacher, training: bool=True):
    """Activate our Autoencoder propagating the input through the Encoder and Decoder respectively

    Arguments:
      input (tensor): Tensor containing the input to the Encoder
      teacher (tensor): Tensor containing the input to the Decoder
      training (bool): Indicates whether regularization methods should be used or not when calling the Autoencoder 

    Returns:
      predictions (tensor): Tensor containing the reconstructed input
    """

    hs_1, hs_2 = self.Encoder(input, training=training)
    predictions = self.Decoder(teacher, states_1=hs_1, states_2=hs_2, training=training)
    return predictions