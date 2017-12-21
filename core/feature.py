# -*- coding: UTF-8 -*-

'''
Created on 2016年12月10日

@author: mull
'''
from numpy import average


'''part of data load function'''

import os
import sys

import numpy as np
from interface import fileProcess


def load_info(filepath=''):
    f = open(filepath)
    line = f.readline()
    doc_terms_list = []
    doc_class_list = []
    while line:
        label = line[line.find(']') + 1:-1]
        if label == '':
            label = line[line.find(']') + 1:]
        content = line[line.find('[') + 1:line.find(']')]
        content = content.split(',')
        doc_terms_list.append(content)
        doc_class_list.append(label)
        line = f.readline()
        
    return doc_terms_list, doc_class_list

def load_term(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
            
    # term set, produce the dictionary by sorted indexes
    term_set_list = sorted(term_set_dict.keys())
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    
    return term_set_dict

def load_class(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    
    return  class_dict

def stats_term_df(doc_terms_list, term_dict):
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
    for term in term_dict:
        for doc_terms in doc_terms_list:
            if term in doc_terms_list:
                term_df_dict[term] += 1
                               
    return term_df_dict

def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
        
    return class_df_list

def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
            
    return  term_class_df_mat

'''part of feature selection function'''
        
def MI(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)
    
    term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size)))
#     term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + 1.0)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)

    f_model = {}
    for i in range(len(term_set)):
        f_model[term_set[i]] = term_score_array[i]

    return Normalize(f_model)

def f_values(doc_terms_list, doc_class_list, fs_method='MI'):
    '''
    @param fs_method: maybe some other function 
    '''
    class_dict = load_class(doc_class_list)
    term_dict = load_term(doc_terms_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)
    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x : x[1])]
    f_model = {}
    
    # reflect from self function
    from core import feature
    f_model = getattr(feature, fs_method)(class_df_list, term_set, term_class_df_mat)
        
    return f_model

'''part of post processing function'''

def Normalize(f_model={}):
#     maxValue = max(f_model.values())
#     minValue = min(f_model.values())
#     for word in f_model:
#         f_model[word] = (1.0 * f_model[word] - minValue) / (maxValue - minValue)

    f_model_s = sorted(f_model.items(), key=lambda item:item[1], reverse=False)
    L = len(f_model_s)
    val = 0.1
    for t in f_model_s:
        val += ((0.9 - 1E-6) / L)
        f_model[t[0]] = val
        
    return f_model

def auto_attention_T(f_model, select_prop=0.02):
    '''
    select the min value of best(select_prop) feature as attention_T
    '''
    sf_model = sorted(f_model.items(), key=lambda item:item[1], reverse=False)
    attention_T = sf_model[int((1 - select_prop) * len(sf_model)) - 1][1]
    
    return attention_T

if __name__ == "__main__":
    
    filepath = fileProcess.auto_config_root() + 'exp_mid_data/sentences_labeled55000.txt'
    doc_terms_list, doc_class_list = load_info(filepath)
    '''input the texts list, classes list, the called method in IG, CHI and MI'''
    f_model = f_values(doc_terms_list, doc_class_list, 'MI')
 
 
    f_model_s = sorted(f_model.items(), key=lambda item:item[1], reverse=False)
#     for i in f_model:
#         print i[0],' ',i[1]
    sf_model = dict(f_model_s[int((1 - 0.2) * len(f_model_s)) - 1 :])
#     print(sf_model)
#     for key in sf_model.keys()[len(sf_model.keys()) - 100 : ]:
#         print type(key), ': ', key, ' ', sf_model[key]
    for key in f_model_s[len(f_model_s) - 800 : ]:
        print type(key[0]), ': ', key[0], ' ', key[1]
    print('max: {0}'.format(max(sf_model.values())))
    print('min: {0}'.format(min(sf_model.values())))
    print(len(f_model_s))
    print(auto_attention_T(f_model, select_prop=0.02))
    
    '''
    test the repeat rate of 3 f_select methods
    '''
    #===========================================================================
    # f_model1 = f_values(doc_terms_list, doc_class_list, 'MI')
    # f_model2 = f_values(doc_terms_list, doc_class_list, 'CHI')
    # f_model3 = f_values(doc_terms_list, doc_class_list, 'IG')
    # f_model_s1 = sorted(f_model1.items(), key=lambda item:item[1], reverse=False)
    # f_model_s2 = sorted(f_model2.items(), key=lambda item:item[1], reverse=False)
    # f_model_s3 = sorted(f_model3.items(), key=lambda item:item[1], reverse=False)
    # 
    # select_prop = 0.02
    # sf_model1 = f_model_s1[int((1 - select_prop) * len(f_model_s1)) - 1 : ]
    # sf_model2 = f_model_s2[int((1 - select_prop) * len(f_model_s2)) - 1 : ]
    # sf_model3 = f_model_s3[int((1 - select_prop) * len(f_model_s3)) - 1 : ]
    # 
    # e_strs1 = set(ele[0] for ele in sf_model1)
    # e_strs2 = set(ele[0] for ele in sf_model2)
    # e_strs3 = set(ele[0] for ele in sf_model3)
    # 
    # rp_12 = e_strs1 & e_strs2
    # rp_13 = e_strs1 & e_strs3
    # rp_23 = e_strs2 & e_strs3
    # rp_123 = (e_strs1 & e_strs2) & e_strs3
    # 
    # rp_rate12 = len(rp_12) * 1.0 / len(sf_model1)
    # rp_rate13 = len(rp_13) * 1.0 / len(sf_model1)
    # rp_rate23 = len(rp_23) * 1.0 / len(sf_model1)
    # rp_rate123 = len(rp_123) * 1.0 / len(sf_model1)
    # 
    # print('total num: {0}'.format(len(sf_model1)))
    # print('rp_12 num: {0}'.format(len(rp_12)))
    # print('rp_13 num: {0}'.format(len(rp_13)))
    # print('rp_23 num: {0}'.format(len(rp_23)))
    # print('rp_123 num: {0}'.format(len(rp_123)))
    # print('rp_rate12: {0}'.format(rp_rate12))
    # print('rp_rate13: {0}'.format(rp_rate13))
    # print('rp_rate23: {0}'.format(rp_rate23))
    # print('rp_rate123: {0}'.format(rp_rate123))
    #===========================================================================
    