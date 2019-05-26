# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:06:55 2019

@author: Ethan
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
from TF2_0_data_preprocessor import TF_preprocessor

class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
# sample_hidden = encoder.initialize_hidden_state()
# sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
# attention_layer = BahdanauAttention(10)
# attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
# decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
# sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
#                                       sample_hidden, sample_output)
class Transformer(tf.keras.Model):

    def __init__(self, checkpoint_dir, tgt_tokenizer, vocab_size_src = 2**13, vocab_size_tgt = 2**13,
                 embedding_dim = 256, enc_units = 1024, dec_units = 1024, learning_rate = 0.2):
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.dec_units = dec_units
        #self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.tgt_tokeinzer = tgt_tokenizer

        self.encoder = Encoder(self.vocab_size_src, self.embedding_dim, self.enc_units, self.batch_size)
        self.decoder = Decoder(self.vocab_size_tgt, self.embedding_dim, self.dec_units, self.batch_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer = self.optimizer,
                                              encoder = self.encoder,
                                              decoder = self.decoder)

    def loss_function(self, real, pred):

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, src_batch, tgt_batch, enc_hidden, batch_size):

        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(src_batch, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tgt_tokeinzer.word_index['<start>']] * batch_size, 1)

            for t in range(1, tgt_batch.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(tgt_batch[:, t], predictions)
                dec_input = tf.expand_dim(tgt_batch[: t], 1)

        batch_loss = (loss / int(tgt_batch.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train_model(self, dataset, steps_per_epoch, batch_size = 1024, epochs = 10, checkpoint = 2):

        start_training = time.time()

        for epoch in epochs:
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (src_batch, tgt_batch)) in enumerate(dataset.take(len(steps_per_epoch))):
                batch_loss = self.train_step(src_batch, tgt_batch, enc_hidden, batch_size)
                total_loss += batch_loss
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

            if (epoch + 1) % checkpoint == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        end_training = time.time()
        hours = (end_training-start_training) // 3600
        min = (end_training - 3600 * hours) // 60
        sec = end_training - 3600 * hours - 60 * min
        print("Total training time: {} hours {} minutes {} seconds\n".format(hours, min, sec))

    def translate_sent(self, sentence):

        input = sentence
        result = ''
        hidden = [tf.zeros((1, self.enc_units))]
        enc_output, enc_hidden = self.encoder(input, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tgt_tokeinzer.word_index['<start>']], 0)


        for t in range(max_len_tgt):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.tgt_tokenizer.index_word[predicted_id] + ' '

            if self.tgt_tokeinzer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def translate(self, tensor, save_file):

        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        start_translate = time.time()
        tensor = tf.convert_to_tensor(tensor)
        with open(save_file, 'w') as f:

            for sent in tensor:
                result = translate_sent(sent)
                f.write("%s\n" % result)
        f.close()
        print('Total translation time: {}'.format(time.time() - start_translate))

if __name__ == '__main__':

    src_train_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_train.de-en.en'
    tgt_train_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_train.de-en.de'
    src_dev_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_val.de-en.en'
    tgt_dev_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_val.de-en.de'
    src_dev_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_test.de-en.en'
    tgt_dev_dir = '/linguistics/ethan/TensorFlow_translation/Training Data from europarl-v7/europarl-v7_test.de-en.de'

    src_lang = 'eng'
    tgt_lang = 'deu'
    vocab_size = 2 ** 13

    tf_p = TF_preprocessing(src_train_dir, tgt_train_dir, src_lang, tgt_lang, vocab_size)
    # tf_p.update_text_tokenizer()

    tra_src_tensor, tra_tgt_tensor = tf_p.load_dataset()
    val_src_tensor, val_tgt_tensor = tf_p.load_val_dataset(src_dev_dir, tgt_dev_dir)
    test_src_tensor, test_tgt_tensor = tf_p.load_val_dataset(src_test_dir, tgt_test_dir)

    src_tokenizer = tf_p.src_tokenizer
    tgt_tokenizer = tf_p.tgt_tokenizer

    # Hparams Assignment
    BUFFER_SIZE = len(tra_src_tensor)
    batch_size = 1024
    steps_per_epoch = len(tra_src_tensor) // batch_size

    dataset = tf.data.Dataset.from_tensor_slices((tra_src_tensor, tra_tgt_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    init_hparams = {
        'embedding_dim': 256,
        'enc_units': 1024,
        'dec_units':1024,
        'learning_rate': 0.2,
        'vocab_size_src': len(src_tokenizer.word_index) + 1,
        'vocab_size_tgt': len(tgt_tokenizer.word_index) + 1,
    }

    train_hparams = {
        'batch_size': 1024,
        'steps_per_epoch': steps_per_epoch,
        'epochs': 10,
        'checkpoint': 5
    }

    checkpoint_dir = ''
    save_file = ''
    TF = Transformer(checkpoint_dir = checkpoint_dir, tgt_tokenizer = tgt_tokenizer, **init_hparams)
    TF.train_model(dataset, **train_hparams)
    TF.translate(test_src_tensor, save_file)