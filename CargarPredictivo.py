import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(42)
import pickle
from keras.models import Sequential, load_model
# from keras.layers import Dense, Activation
# from keras.layers import LSTM, Dropout
# from keras.layers import TimeDistributed
# from keras.layers.core import Dense, Activation, Dropout, RepeatVector
# from keras.optimizers import RMSprop
# from keras.models import Sequential

import matplotlib.pyplot as plt
import sys
from pylab import rcParams
import seaborn as sns

SEQUENCE_LENGTH = 40
path = 'TextoPrueba.txt'
text = open(path, encoding='utf-8').read().lower()
chars = sorted(list(set(text)))


model = load_model('keras_model.h5')
history = pickle.load(open("history.p", "rb"))

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
        
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)