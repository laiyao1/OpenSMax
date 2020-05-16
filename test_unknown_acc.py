import os
import copy
import pickle

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from keras.models import Model
from keras.models import load_model
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from collections import Counter


import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_av(model_name, X_train, y_train,  model, unknown_id, method):
    class_num = y_train.shape[-1]
    res = []
    for i in range(class_num):
        res.append([])
    y_predict = np.argmax(model.predict(X_train), axis = 1)
    y_true = np.argmax(y_train, axis = 1)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('mav').output)
    openmax_train = intermediate_layer_model.predict(X_train)
    softmax_train = model.predict(X_train)
    for i in range(X_train.shape[0]):
        if y_predict[i] == y_true[i]:
            openmax_av = openmax_train[i]
            softmax_av = softmax_train[i]
            label = y_true[i]
            if method == 'svm1':
                res[label].append([openmax_av[label]])
            elif method == 'svm2':
                res[label].append(np.array([openmax_av[label], softmax_av[label]]))
            elif method == 'svmn' or method == 'openmax':
                res[label].append(openmax_av)
            elif method == 'svm2n':
                res[label].append(np.concatenate(openmax_av, openmax_av))
            elif 'doc' in method:
                res[label].append(1+(1-softmax_av[label]))
    for i in range(class_num):
        res[i] = np.array(res[i])
        print('res[i].shape', i, res[i].shape)
    mav = []
    vav = []
    for i in range(class_num):
        mav.append([np.mean(res[i],axis = 0)])
        vav.append([np.var(res[i],axis = 0)])
    print('mav', mav)
    np.savez(model_name + '_mav_av_data.npz',mav = mav, av = res)
    print('mav_av_data has been saved')
    return mav, res, vav

    
# One class SVM
# av: activation values
# nu: the hyperparameter
def outlier_detector(av, method = 'svm2', parameter = 0.05):
    clf = []
    scaler = []
    for i in range(len(av)):
        if len(av[i]) <= 10:
            clf.append(None)
            scaler.append(None)
        else:
            av_t = np.array(av[i])
            if method == 'svm2' or method == 'svmn' or method == 'svm2n' or method == 'svm1':
                clf_t = svm.OneClassSVM(nu= parameter, kernel="linear")
            else:
                clf_t = IsolationForest(n_estimators=10)
            av_t = av_t.reshape((-1, av_t.shape[-1]))
            scaler_t = MinMaxScaler()
            scaler_t.fit(av_t)
            print('av_t shape =', scaler_t.transform(av_t).shape)
            # print('av_t =', av_t)
            # print(scaler_t.transform(av_t))
            if method == 'svmn' or method == 'svm2n':
                clf_t.fit(av_t)
            else:
                clf_t.fit(scaler_t.transform(av_t))
            clf.append(clf_t)
            scaler.append(scaler_t)
    return clf, scaler

    
def draw_confusion_matrix_image(cm, filename = "confusion_matrix.png"):
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", vmin=0, vmax = 1600)
    ax.set_xlabel("Predict Labels", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    plt.savefig(filename, dpi=600, format = 'pdf')


# suitable method: 'doc', 'doc0.5', 'svm1', 'svm2'
# openmax_test: output of openmax layer
# softmax_test: output of softmax layer
# clf: the classify function of One Class SVM

def calculate_known_acc(model_name, X_test, y_test, model, clf, unknown_id, method = 'svm2', isdraw = True, mav = None, vav = None ,alpha = 3, scaler = None, p = 0.1):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('mav').output)
    class_num = y_test.shape[-1]
    y_predict = np.argmax(model.predict(X_test), axis = -1)
    y_true = np.argmax(y_test, axis = -1)
    openmax_test = intermediate_layer_model.predict(X_test)
    softmax_test = model.predict(X_test)
    
    profit = 0
    wrong_cnt = 0
    wrong_cnt_correct = 0
    correct_cnt = 0
    correct_cnt_wrong = 0
    cnt_dict = {}
    y_predict_new = copy.deepcopy(y_predict)
    
    for i in range(class_num):
        cnt_dict[i] = 0 
    for i in range(X_test.shape[0]):
        label = y_predict[i]
        if method == 'doc':
            is_unknown = softmax_test[i][label] < 0.5 or softmax_test[i][label] < mav[label] - alpha * vav[label]
        elif clf[label] != None:
            if method == 'percentile':
                is_unknown = openmax_test[i][label] < clf[label]
            elif method == 'svm1':
                is_unknown = clf[label].predict(openmax_test[i][label].reshape(1,1))[0] == -1
            elif method == 'svm2' or method == 'svm2_av':
                tmp = np.array([openmax_test[i][label], softmax_test[i][label]]).reshape(1,2)
                tmp = scaler[label].transform(tmp)
                if method == 'svm2_av':
                    is_unknown = ((openmax_test[i][label].reshape(1,1) < mav[label][0]  \
                                and softmax_test[i][label].reshape(1,1) < mav[label][1]  ) \
                                and clf[label].predict(tmp)[0] == -1)
                else:
                    is_unknown = (clf[label].predict(tmp)[0] == -1)
            elif method == 'svmn':
                tmp = np.array(openmax_test[i]).reshape(1,class_num)
                is_unknown = clf[label].predict(tmp)[0] == -1
            elif method == 'svm2n':
                tmp = np.concatenate((np.array(openmax_test[i]), np.array(softmax_test[i]))).reshape(1,-1)
                is_unknown = clf[label].predict(tmp)[0] == -1
            else:
                is_unknown = clf[label].predict(np.array([openmax_test[i][label], softmax_test[i][label]]).reshape(1,2))[0] == -1
        if y_predict[i] == y_true[i]:
            correct_cnt += 1
            if (clf == None or clf[label] != None) and is_unknown:
                correct_cnt_wrong += 1
                # print('make the correct data unknown')
                profit -= 1
                y_predict_new[i] = unknown_id
            else:
                pass
                # print('still correct')
        else:
            wrong_cnt += 1
            if (clf == None or clf[label] != None) and is_unknown:
                # print('make the wrong data unknown')
                profit += 1
                wrong_cnt_correct += 1
                cnt_dict[y_true[i]] += 1
                y_predict_new[i] = unknown_id
            else:
                pass
    print('unknown_id', unknown_id)
    cond_predict_new = (y_predict_new == unknown_id)
    cond_true = (y_true == unknown_id)
    z_predict_new = np.where(cond_predict_new,1,0)
    z_true = np.where(cond_true,1,0)
    
    # print('profit', profit)
    # print('wrong_cnt',wrong_cnt)
    # print('wrong_cnt_correct',wrong_cnt_correct)
    # print('rate',wrong_cnt_correct/wrong_cnt)
    # print('correct_cnt',correct_cnt)
    # print('correct_cnt_wrong',correct_cnt_wrong)
    # print('rate',correct_cnt_wrong/correct_cnt)
    # print('cnt_dict',cnt_dict)
    
    print('### without known detection ###')
    print('confusion_matrix:')
    print(sklearn.metrics.confusion_matrix(y_true,y_predict))
    print('acc(before unknown method) =',sklearn.metrics.accuracy_score(y_true,y_predict))
    if isdraw:
        draw_confusion_matrix_image(sklearn.metrics.confusion_matrix(y_true,y_predict), "confusion_matrix_"+model_name+ "_before.pdf")
    print('m_f1score(before unknown method) =',sklearn.metrics.f1_score(y_true, y_predict, average='macro'))
    print('### with known detection ###')
    print('confusion_matrix:')
    print(sklearn.metrics.confusion_matrix(y_true,y_predict_new))
    acc = sklearn.metrics.accuracy_score(y_true,y_predict_new)
    print('acc(after unknown method)  =', acc)
    if isdraw:
        draw_confusion_matrix_image(sklearn.metrics.confusion_matrix(y_true,y_predict_new), "confusion_matrix_add_unknown_"+model_name+"_" + str(p)+ "_after.pdf")
    print('precision = ',sklearn.metrics.precision_score(z_true, z_predict_new))
    print('recall = ',sklearn.metrics.recall_score(z_true, z_predict_new))
    f1score = sklearn.metrics.f1_score(z_true, z_predict_new,average='macro')
    print('macro f1score = ',f1score)
    m_f1score = sklearn.metrics.f1_score(y_true, y_predict_new,average='macro')
    print('multi macro f1score = ',m_f1score)
    return m_f1score, acc

    

# X_train, y_train, X_test, y_test: evaulation data
# model_name: the model used in known detection phase
# filepath: the path of known detection model weights file 
# method: the method used in unknown detection
# add_av: whether to add mean value restrictive constrains
def get_unknown_acc(model_name, X_train, y_train, X_test, y_test, filepath, method = 'svm2'):
    print('method =', method)
    model = load_model(filepath)
    class_num = y_train.shape[-1]
    y_true = np.argmax(y_train, axis = 1)
    hash = [0 for i in range(class_num)]
    for y in y_true:
        hash[y] = 1
    for i in range(class_num):
        if hash[i] == 0:
            unknown_id = i
            break
    print('unknown_id =', unknown_id)
    # def get_av(model_name, X_train, y_train,  model, unknown_id, method)
    mav, av, vav = get_av(model_name, X_train, y_train, model, unknown_id, method)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('mav').output)
    best_f1 = 0.0
    best_acc = 0.0
    best_nu = 0
    res_f1 = []
    res_acc = []
    for p in [0.5, 0.075, 0.1, 0.125, 0.15]:
        print('p =', p)
        if method == 'doc':
            #calculate_known_acc(model_name, X_test, y_test, model, clf, unknown_id, method = 'svm2', isdraw = True, mav = None, vav = None ,alpha = 3, scaler = None, p = 0)
            f1_score, acc = calculate_known_acc(model_name, X_test, y_test, model, None, unknown_id, method, False, mav, vav, p = p*10000)
        else:
            clf, scaler = outlier_detector(av, method = method, parameter = p)
            f1_score, acc = calculate_known_acc(model_name, X_test, y_test, model, clf, unknown_id, method, False, mav, vav, scaler = scaler, p = p)
        if f1_score > best_f1:
            best_f1, best_f1_nu = f1_score, p
        if acc > best_acc:
            best_acc, best_acc_nu = acc, p
        res_f1.append(f1_score)
        res_acc.append(acc)
    print('best_f1 =' ,best_f1, 'best_acc =', best_acc, 'best_f1_nu =', best_f1_nu, ' best_acc_nu = ', best_acc_nu)
    return res_f1, res_acc
    

# Parameter Description:
# model_method: the model of known detection phase
# method_list:  the method of unknown detection phase 
#               'svm1'(OpenSMax-1-1D), 'svmn'(OpenSMax-1-(N+1)D), 'svm2n'(OpenSMax-2-(N+1)D), 'svm2'(OpenSMax-2-1D)
#               'svm2_av' (OpenSMax-2-1D+MVRC), 'doc' (DOC), 'percentile' (Only use the lowerbound)
# draw_line_chart: whether to draw Figure 4                
# output_file_name: output the data for Figure 4

def work(model_name, method_list, draw_line_chart = False, output_file_name = 'line_chart.pkl'):
    data_filepath = model_name + '.npz'
    data = np.load(data_filepath)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    draw_line_chart = False
    if draw_line_chart == False:
        for method in method_list:
            get_unknown_acc(model_name, X_train, y_train, X_test, y_test, filepath = model_name + '_weights.hdf5', method = method)
    else:
        res = {}
        for method in ['svm2']:
            res[method] = {}
            res[method]['f1'], res[method]['acc'] = \
                get_unknown_acc(model_name, X_train, y_train, X_test, y_test, filepath = model_name + '_weights.hdf5', method = 'svm2')
        print(res)
        fopen = open(output_file_name, 'wb')
        pickle.dump(res, fopen)


if __name__ == '__main__':
    work(model_name = 'lstm_top_onehot', method_list = ['svm2'])
        
    