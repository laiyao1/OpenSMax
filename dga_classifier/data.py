"""Generates data for train/test algorithms"""
from datetime import datetime
import io
from urllib.request import urlopen
from zipfile import ZipFile

import pickle
import os
import random
import tldextract


# Location of Alexa 1M
ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

# Our output file containg all the training data
DATA_FILE = 'traindata.pkl'

def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):
    """Grabs Alexa 1M"""
    fopen = open(filename,'r')
    lines = fopen.read().split()[:num]
    res = [tldextract.extract(x.split(',')[1]).domain for x in lines]
    tops = [tldextract.extract(x.split(',')[1]).suffix for x in lines]
    print("get alexa finished")
    fopen.close()
    return res, tops

    
def gen_malicious(num_per_dga = 10000, filename = 'dga.csv'):
    num = 50000
    max_per_catagory = 4000
    cata_dict = {}
    fopen = open(filename,'r')
    domains = []
    labels = []
    tops = []
    inputs = fopen.read().split()[:]
    fopen.close()
    
    # make sure that each catagory would not be exceed max_per_catagory
    inputs_tmp = []
    cata_cnt = {}
    for x in inputs:
        cata = x.split(',')[0]
        if not cata in cata_cnt:
            cata_cnt[cata] = 1
        else:
            cata_cnt[cata] += 1
    print('cata_cnt', cata_cnt)
    for x in inputs:
        label = x.split(',')[0]
        if cata_cnt[label] < 1000:
            continue
        if not label in cata_dict:
            cata_dict[label] = 1
        else:
            if cata_dict[label] >= max_per_catagory:
                continue
            else:
                cata_dict[label] += 1
        inputs_tmp.append(x)
    inputs = inputs_tmp
    random.seed(1)
    random.shuffle(inputs)
    for x in inputs[:]:
        if tldextract.extract(x.split(',')[1]).subdomain != '':
            domain = tldextract.extract(x.split(',')[1]).subdomain
        else:
            domain = tldextract.extract(x.split(',')[1]).domain
        domains.append(domain)
        labels.append(x.split(',')[0])
        top = tldextract.extract(x.split(',')[1]).suffix
        tops.append(top)
    print('domains top 100',domains[:100])
    print('labels top 100',labels[:100])
    print('dga domain number =',len(domains))
    cata_cnt = {}
    for cata in labels:
        if not cata in cata_cnt:
            cata_cnt[cata] = 1
        else:
            cata_cnt[cata] += 1
    print('cata_dict', cata_dict)
    return domains, labels, tops
    
    
def gen_data(force=False):
    """Grab all data for train/test and save

    force:If true overwrite, else skip if file
          already exists
    """
    if force or (not os.path.isfile(DATA_FILE)):
        # 删除某类
        domains, labels,tops = gen_malicious(10000, filename = 'dga.csv')

        # Get equal number of benign/malicious
        benign_num = 2000
        domains_b, tops_b = get_alexa(benign_num)
        domains += domains_b
        labels += ['benign']*benign_num
        tops += tops_b
        # print(labels)
        # print(domains)
        pickle.dump(zip(labels, domains, tops), open(DATA_FILE, 'wb'))
        print('write file finished')
        
        class_dict = {'benign': 0}
        class_names = set()
        for label in labels:
            if label != 'benign':
                class_names.add(label)
        words = sorted(list(set(class_names)))
        cnt = 1 
        for word in words:
            class_dict[word] = cnt
            cnt += 1
            
        print(class_dict)
        
        fopen = open('class_dict.pkl', 'wb')
        pickle.dump(class_dict,fopen)
        pickle.dump(cnt, fopen)
        print('write file(class_dict.pkl) finished')
        
        
def get_data(force=False):
    """Returns data and labels"""
    gen_data(force)
    return list(pickle.load(open(DATA_FILE,'rb')))

    
if __name__ == '__main__':
    get_data(force = True)