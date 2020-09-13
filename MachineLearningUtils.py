import pandas as pd
from collections import OrderedDict
import numpy as np

def bag_of_words(df,n_words):
    
    ''' Reads in a dataframe (single column), and n_words. 
    Returns a one hot encoded feature set 
    with length equal to n_words'''
    
    items = df.values.tolist()
    word_d = {}
    for item in items:
        try:
            item = item.split(' ')
            for i in item:
                if i in word_d:
                    word_d[i]+=1
                else:
                    word_d[i]=1
        except:
            pass
        
    word_d = {k: v for k, v in sorted(word_d.items(), key=lambda item: item[1], reverse=True)}
    word_d = dict(list(word_d.items())[:n_words])
    keys = list(word_d.keys())

    word_vecs = []
    for item in items:
        if item == item:
            item = item.split(' ')
            word_vec = [0 for x in range(len(keys))]
            for it in item:
                for i, word in enumerate(word_d):
                    if it == word:
                        word_vec[i] = 1
            word_vecs.append(word_vec)

        else:
            word_vecs.append([0 for x in range(len(keys))])
    
    
    return keys, word_vecs

def encode_titles(df,col,titles = ["Mrs","Mr","Misses","Ms"]):
    '''One hot encodes given titles on an input dataframe and column. Returns a list of one hot encoded vectors'''
    encodings = []
    for _,r in df.iterrows():
        name = r[col]
        one_hot_vec = [0 for x in range(len(titles))]
        for i,t in enumerate(titles):
            if t in name:
                one_hot_vec[i]=1

        encodings.append(one_hot_vec)
                
    return encodings