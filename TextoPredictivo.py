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
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams

quotes = ['y por eso es que me encuentro en esta',
         'mañana tengo que ir a trabajar al banco a las',
         'Tengo que estar en la tarde en el centro comercial para',
         'pero tengo que esperar a mi padre en la parada de buses hasta',
         'tengo demasiado sueño y me gustaría estar haciendo cualquier']

path = 'TextoPrueba.txt'
text = open(path, encoding='utf-8').read().lower()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

SEQUENCE_LENGTH = 40
# step = 2
# sentences = []
# next_chars = []
# for i in range(0, len(text) - SEQUENCE_LENGTH, step):
#     sentences.append(text[i: i + SEQUENCE_LENGTH])
#     next_chars.append(text[i + SEQUENCE_LENGTH])
# print(f'num training examples: {len(sentences)}')

# X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         X[i, t, char_indices[char]] = 1
#     y[i, char_indices[next_chars[i]]] = 1

# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history

# model.save('keras_model.h5')
# pickle.dump(history, open("history.p", "wb"))

model = load_model('keras_model.h5')
history = pickle.load(open("history.p", "rb"))

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
    return x

def sample(preds, top_n=5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    print (original_text)
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        print ('original_text: ', original_text +'\nLenght: ', len(original_text))
        print()
        print('completion: ', completion +'\nLenght: ', len(completion))
        input()
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

print(predict_completions('hol'))