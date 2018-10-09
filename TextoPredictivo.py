#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from keras.models import Sequential

import matplotlib.pyplot as plt
import sys
from pylab import rcParams
import seaborn as sns

path = sys.arg[1]
text = open(path, encoding='utf-8').read().lower()
print('Length: ', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')

SEQUENCE_LENGTH = 40
step = 2
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Sentence: ', sentences[0])
print('Siguiente: ', next_chars[0])
print(X.shape)
print(y.shape)

#model = Sequential()
#model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
#model.add(Dense(len(chars)))
#model.add(Activation('softmax'))

#optimizer = RMSprop(lr=0.01)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

#model.save('keras_model.h5')
#pickle.dump(history, open("history.p", "wb"))
