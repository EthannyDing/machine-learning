# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:40:35 2019

@author: Ethan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string 
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
pd.set_option('display.max_colwidth', 200)

def read_text(filename):
    # open the file
    file = open(filename, mode='rt', encoding='utf-8')
    -
    # read all text
    text = file.read()
    file.close()
    return text

def to_lines(text):
      sents = text.strip().split('\n')
      sents = [i.split('\t') for i in sents]
      return sents

data = read_text("english2german.txt")
deu_eng = to_lines(data)
deu_eng = np.array(deu_eng)

train = deu_eng[:50000, :]
train[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in train[:,0]]
train[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in train[:,1]]

for i in range(len(train)):
    train[i,0] = train[i,0].lower()
    train[i,1] = train[i,1].lower()

# empty lists
eng_l = []
deu_l = []

# populate the lists with sentence lengths
for i in train[:,0]:
    eng_l.append(len(i.split()))

for i in train[:,1]:
    deu_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'deu':deu_l})

length_df.hist(bins = 30)
plt.show()

# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# prepare english tokenizer
eng_tokenizer = tokenization(train[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = np.max(eng_l)
print('English Vocabulary Size: %d' % eng_vocab_size)

# prepare Deutch tokenizer
deu_tokenizer = tokenization(train[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1

deu_length = np.max(deu_l)
print('Deutch Vocabulary Size: %d' % deu_vocab_size)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

# prepare training data
X = encode_sequences(deu_tokenizer, deu_length, train[:, 1])

from sklearn.model_selection import train_test_split
train_x, test_x, train_labels, test_labels = train_test_split(X, train[:, 0], test_size=0.2, random_state = 12)

train_y = encode_sequences(eng_tokenizer, eng_length, train_labels)
test_y = encode_sequences(eng_tokenizer, eng_length, test_labels)

# build NMT model
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

model = build_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)
rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')
model.summary()

filename = 'NMT.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(train_x, train_y.reshape(train_y.shape[0], train_y.shape[1], 1), 
          epochs=20, batch_size=512, 
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)


model = load_model('NMT.h5')
preds = model.predict_classes(test_x.reshape((test_x.shape[0],test_x.shape[1])))

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
             
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))
    
pred_df = pd.DataFrame({'actual' : test_labels, 'predicted' : preds_text})
pd.set_option('display.max_colwidth', 200)

pred_df.sample(15)
