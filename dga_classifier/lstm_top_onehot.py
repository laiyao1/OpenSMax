"""Train and test lstm_top_onehot classifier"""
# ATTENTION: this model deleted the bigram and manual feature.
import os
import random

import dga_classifier.data as data

from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Concatenate, Lambda
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Model
from keras.utils.np_utils import to_categorical
import tensorflow as tf

import sklearn
from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import dga_classifier.split as split
import pickle

from collections import Counter

method = "lstm_top_onehot"
filepath =  method +'_weights.hdf5'

def build_model(char_feature_dimension, tld_feature_dimension, max_seq_len, multi_class = False, class_num = 2):
    # [0:max_seq_len] -> raw string
    # [max_seq_len:] -> tld_feature_dimension
    print('char_feature_dimension', char_feature_dimension)
    print('tld_feature_dimension', tld_feature_dimension)
    print('max_seq_len', max_seq_len)
    inputs = Input(shape=(max_seq_len + tld_feature_dimension,))
    x1 = Lambda(lambda x: x[:,: max_seq_len], name = "Lambda_tmp")(inputs) # raw string
    x2 = Lambda(lambda x: x[:, max_seq_len:], name = "Lambda_tmp2")(inputs) # tld one hot
    x = Embedding(input_dim = char_feature_dimension, output_dim = 128, input_length=max_seq_len)(x1)
    x = LSTM(512)(x)
    x = Dropout(0.8)(x)
    
    mid = Dense(128, input_dim=char_feature_dimension, activation='relu')(x2)
    mid = Dropout(0.8)(mid)
    mid2 = Dense(128,  activation='relu')(mid)
    mid2 = Dropout(0.8)(mid2)
    
    x = Concatenate(axis=-1)([mid2, x])
    if not multi_class:
        predictions = Dense(1, input_dim = 1024, activation = 'sigmoid')(x)
        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs = inputs, outputs = predictions)
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    else:
        x = Dense(class_num, input_dim=1024 , name = 'mav')(x)
        predictions = Activation('softmax')(x)
        model = Model(inputs = inputs, outputs = predictions)
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model
    
# ATTENTION: the choice of data_cache 
def run(max_epoch=50, batch_size=128, cata_split = True, multi_class = False, ratio = None, data_cache = False):
    global filepath
    """Run train/test on logistic regression model"""
    print('method is LSTM(SLD) + One-Hot(TLD)')
    indata = data.get_data()

    # Extract data and labels
    randnum = 1
    random.seed(randnum)
    random.shuffle(indata)
    X = [x[1] for x in indata]
    labels = [x[0] for x in indata]
    tops = [x[2] for x in indata]
    
    # One hot top domain one-hot features
    le = LabelEncoder()
    new_tops = le.fit_transform(tops)
    new_tops = new_tops.reshape(-1, 1)
    top_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    top_enc.fit(new_tops)
    feature_top = top_enc.transform(new_tops)
    tld_feature_dimension = feature_top.shape[1]
    print('tld_feature_dimension(the number of top domain feature)', tld_feature_dimension)
    
    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
    char_feature_dimension = len(valid_chars) + 1
    max_seq_len = np.max([len(x) for x in X])
    
    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=max_seq_len)
    print('X shape',X.shape)
    
    # Convert labels to 0-1
    if not multi_class:
        y = [0 if x == 'benign' else 1 for x in labels]
        class_num = 2
    else:
        DATA_FILE = 'class_dict.pkl'
        fopen =open(DATA_FILE,'rb')
        label_dict = pickle.load(fopen)
        print('label_dict', label_dict)
        class_num = pickle.load(fopen)
        print('class_num', class_num)
        y = []
        if ratio != None:
            max_class_num = int(class_num * ratio) + 1
            print('max_class_num', max_class_num)
            for x in labels:
                if label_dict[x] > class_num * ratio:
                    y.append(max_class_num)
                else:
                    y.append(label_dict[x])
        else:
            y = [label_dict[x] for x in labels]
        print('y',y[:100])
        print('first_ y',Counter(y))
        y = to_categorical(y, num_classes = class_num)

    # concatenate all the features
    X = np.concatenate(( X, feature_top),axis = -1)
    print('X shape(after feature concatenate)', X.shape)
    
    final_data = []
    print('cata_split = ',cata_split)
    fwrite = open('bigram_lstm_fe.log','w')
    
    if data_cache == True and os.path.isfile('bigram_lstm_fe_top_data.npz'):
        np_data = np.load('bigram_lstm_fe_top_data.npz')
        X_train = np_data['X_train']
        y_train = np_data['y_train']
        X_test = np_data['X_test']
        y_test = np_data['y_test']
        label_test = np_data['label_test']
    else:
        if cata_split == False:
            print('cata split is false')
            X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                       test_size=0.2, random_state= 1)
        else:
            if ratio != None:
                X_train, X_test, y_train, y_test, label_train, label_test = split.train_test_split_as_catagory(X, y, labels, cata = max_class_num)
            else:
                X_train, X_test, y_train, y_test, label_train, label_test = split.train_test_split_as_catagory(X, y, labels, cata = 'symmi')
        print('X_train shape',X_train.shape)
        print('X shape',X.shape)
        np.savez('bigram_lstm_fe_top_data.npz',X_train = X_train, y_train= y_train,
                                X_test = X_test, y_test = y_test, label_test = label_test)
        print('bigram_lstm_fe_top_data.npz has been saved')
    
    print('type X_train',type(X_train))
    print('Build model...')
    model = build_model(char_feature_dimension, tld_feature_dimension, max_seq_len, multi_class, class_num)
    print("Train...")
    best_iter = -1
    best_acc = 0.0

    for ep in range(max_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1)
        if not multi_class:
            t_probs = model.predict_proba(X_test)
            t_acc = sklearn.metrics.accuracy_score(y_test, t_probs> .5)
            print('Epoch %d: acc = %f (best=%f)' % (ep, t_acc, best_acc))
            fwrite.write('Epoch %d: acc = %f (best=%f)\n' % (ep, t_acc, best_acc))
            probs = t_probs
            print('test confusion matrix')
            print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
        else:
            score = model.evaluate(X_test, y_test, verbose = 0)
            t_acc = score[1]
            print('Epoch %d: acc = %f (best=%f)' % (ep, t_acc, best_acc))
            fwrite.write('Epoch %d: acc = %f (best=%f)\n' % (ep, t_acc, best_acc))
        if t_acc > best_acc:
            best_acc = t_acc
            best_iter = ep
            model.save(filepath)
            opt_model = model
            print('newest model has been saved') 
    fwrite.close()