import os
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import Counter
ch_cata = None


def extract_unknown_class(X, y, labels, cata =None):
    global ch_cata
    ch_cata = cata
    if ch_cata is None:
        while True:
            index = random.randint(0, len(labels)-1)
            ch_cata = labels[index]
            if not ch_cata == 'benign':
                break
    print('In this fold, we choose ', ch_cata, ' as the test label.')
    X = X.tolist()
    y_true = np.argmax(y, axis = 1)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    label_train = []
    label_test = []
    cnt= 0
    for i in range(len(X)):
        if labels[i] == ch_cata:
            cnt += 1
    tmp = 0
    cnt_benign = 0
    y_true = np.argmax(y, axis = 1)
    for i in range(len(X)):
        if ((isinstance(ch_cata,str) and labels[i] == ch_cata) or \
            (isinstance(ch_cata,int) and y[i][ch_cata] == 1))  :
            if len(y_test) < 3000:
                X_test.append(X[i])
                y_test.append(y[i])
                label_test.append(labels[i])
        else:
            if (isinstance(ch_cata,int) and y[i][0] == 1):
                if cnt_benign >=8000:
                    continue
                else:
                    cnt_benign += 1
            X_train.append(X[i])
            y_train.append(y[i])
            label_train.append(labels[i])
    y_true = np.argmax(y_train, axis = 1)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test, label_train, label_test
            

def train_test_split_as_catagory(X, y, labels, cata = None, ratio = 0.2):
    X_train, X_test, y_train, y_test, label_train, label_test = extract_unknown_class(X, y, labels, cata = cata)
    y_true = np.argmax(y_train, axis = 1)
    y_test_true = np.argmax(y_test, axis = 1)
    X_train, X_tmp, y_train, y_tmp, label_train, label_tmp = train_test_split(X_train, y_train, label_train, 
                                                                           test_size=ratio, random_state= 1)
    y_true = np.argmax(y_train, axis = 1)
    X_test = np.concatenate((X_test, X_tmp), axis = 0 )
    y_test = np.concatenate((y_test, y_tmp), axis = 0)
    y_test_true = np.argmax(y_test, axis = 1)
    y_train_true = np.argmax(y_train, axis = 1)
    label_test += label_tmp
    return X_train, X_test, y_train, y_test, label_train, label_test

            
if __name__ == '__main__':
    X = ['emeewaiq', 'mqmiqsymiidufbtd', 'wwrinqegydvselov', 'google']
    y = [1, 1, 1, 0]
    labels = ['pykspa_v1', 'emotet', 'emotet', 'benign']
    X_train, X_test, y_train, y_test, _, label_test = \
        train_test_split_as_catagory(X, y, labels)
    print('X_train', X_train)
    print('X_test', X_test)
    print('y_train', y_train)
    print('y_test', y_test)
    print('label_test', label_test)