# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:12:21 2019

@author: Ethan
"""

#!pip install tensorflow-gpu==2.0.0-alpha0
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

import numpy as np
import os
import io
from pymoses.lib.preprocessor import Preprocessor


# ***CONSIDER BUILDING A CLASS FOR LANGUAGE PREPROCESSING

class TF_preprocessing(object):
    
    def __init__(self, src_dir, tgt_dir, src_lang, tgt_lang, vocab_size):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.vocab_size = vocab_size

        self.src_text = self.create_datasets(self.src_dir, self.src_lang)
        self.tgt_text = self.create_datasets(self.tgt_dir, self.tgt_lang)
        
        self.src_tokenizer = self.tokenize(self.src_text)
        self.tgt_tokenizer = self.tokenize(self.tgt_text)
        
        self.src_max_len = np.max([len(sent.split(' ')) for sent in self.src_text])
        self.tgt_max_len = np.max([len(sent.split(' ')) for sent in self.tgt_text])
      
    def preprocess_sent(self, sent, lang):

        pp = Preprocessor(lang)
        w = pp.preprocess(sent, 
                          correctCase=False, 
                          correctSpelling=False,
                          applyFilter=True, 
                          tokenize=True, 
                          applyBpe=False)
        w = w.rstrip().strip()
        w = '<start> ' + w + ' <end>'
        return w
    
    def create_datasets(self, path, lang):
        
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        #tgt_lines = io.open(self.tgt_dir, encoding='UTF-8').read().strip().split('\n')
        text = [self.preprocess_sent(line, lang) for line in lines]
        #self.tgt_text = [self.preprocess_sent(tgt, self.tgt_lang) for tgt in tgt_lines]
        return text

    
    def tokenize(self, text):
        
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_word = self.vocab_size, filters='')
        lang_tokenizer.fit_on_texts(text)
        return lang_tokenizer
    
    def sequence_padding(self, text, src = True):
    
        #lang_tokenizer = tokenize(text)
        if src:
            
            tensor = self.src_tokenizer.texts_to_sequences(text)
            maxlen = self.src_max_len
            tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen, padding='post')
        else:
            
            tensor = self.tgt_tokenizer.texts_to_sequences(text)
            maxlen = self.tgt_max_len
            tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen, padding='post')            
        
        return tensor

    def load_dataset(self):
    
        src_tensor = self.sequence_padding(self.src_text, src = True)
        tgt_tensor = self.sequence_padding(self.tgt_text, src = False)
        return src_tensor, tgt_tensor
    
    def load_val_dataset(self, src_dev_dir, tgt_dev_dir):
        
        src_text = self.create_datasets(src_dev_dir, self.src_lang)
        tgt_text = self.create_datasets(tgt_dev_dir, self.tgt_lang)
        
        src_tensor = self.sequence_padding(src_text, src = True)
        tgt_tensor = self.sequence_padding(tgt_text, src = False)
        return src_tensor, tgt_tensor    
    
if __name__ == '__main__':
    
    src_train_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_train.de-en.en'
    tgt_train_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_train.de-en.de'
    src_dev_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_val.de-en.en'
    tgt_dev_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_val.de-en.de'
    
    src_lang = 'eng'
    tgt_lang = 'deu'
    vocab_size = 2**13
 
    tf_p = TF_preprocessing(src_train_dir, tgt_train_dir, src_lang, tgt_lang, vocab_size)
    #tf_p.update_text_tokenizer()
    
    tra_src_tensor, tra_tgt_tensor = tf_p.load_dataset()
    val_src_tensor, val_tgt_tensor = tf_p.load_val_dataset(src_dev_dir, tgt_dev_dir)
    
    print('train source shape : {}'.format(tra_src_tensor.shape()))
    print('train target shape : {}'.format(tra_tgt_tensor.shape()))
    print('validation source shape : {}'.format(val_src_tensor.shape()))
    print('validation target shape : {}'.format(val_tgt_tensor.shape()))
    print("***************")








