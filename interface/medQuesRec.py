# -*- coding: UTF-8 -*-

'''
Created on 2016年11月21日

@author: superhy
'''
import time

import numpy

from core import layer, feature
from core import prencoder
from interface import fileProcess
from interface.embedding import word2Vec


def loadGensimMatData(trainTestFileTuples, gensimW2VModelPath, nb_classes):
    
    fr_train = open(trainTestFileTuples[0], 'r')
    fr_test = open(trainTestFileTuples[1], 'r')
    trainLines = fr_train.readlines()
    testLines = fr_test.readlines()
    fr_train.close()
    fr_test.close()
    del(fr_train)
    del(fr_test)
    
    
    start_load = time.clock()
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    vector_dim = gensimW2VModel.vector_size
    max_len = 0
    for line in trainLines + testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        max_len = len(words) if len(words) > max_len else max_len
    
    trainMatList = []
    trainLabelList = []
    for line in trainLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = numpy.zeros([max_len, vector_dim])
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
#                 lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
                vector = word2Vec.getWordVec(gensimW2VModel, words[i])
                lineVecs[i] = numpy.asarray(vector, dtype='float32')
        trainMatList.append(lineVecs)
        
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        trainLabelList.append(classesVec)
    del(trainLines)
        
    testMatList = []
    testLabelList = []
    for line in testLines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        lineVecs = numpy.zeros([max_len, vector_dim])
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
#                 lineVecs.append(word2Vec.getWordVec(gensimW2VModel, word))
                vector = word2Vec.getWordVec(gensimW2VModel, words[i])
                lineVecs[i] = numpy.asarray(vector, dtype='float32')
        testMatList.append(lineVecs)
        
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        testLabelList.append(classesVec)
    del(testLines)
    
    del(gensimW2VModel)
    
    x_train = numpy.asarray(trainMatList)
    del(trainMatList)
    y_train = numpy.asarray(trainLabelList)
    del(trainLabelList)
    x_test = numpy.asarray(testMatList)
    del(testMatList)
    y_test = numpy.asarray(testLabelList)
    del(testLabelList)
    xy_data = (x_train, y_train, x_test, y_test)
    end_load = time.clock()
    print('finish load train and test numpy array data in {0}s'.format(end_load - start_load))
    
    input_shape = (max_len, vector_dim)
    
    return xy_data, input_shape

def initAttentionValues(t_filepath):
    
    doc_terms_list, doc_class_list = feature.load_info(t_filepath)
    # input the texts list, classes list, called methods: MI
    f_model = feature.f_values(doc_terms_list, doc_class_list)
    
    auto_attention_T = feature.auto_attention_T(f_model)
    
    return f_model, auto_attention_T

def loadAttentionGensimMatData(trainTestFileTuples, gensimW2VModelPath, nb_classes, t_filepath):
    '''
    @param @t_filepath: original corpus file path which contain all training text sentences
    '''
    
    fr_train = open(trainTestFileTuples[0], 'r')
    fr_test = open(trainTestFileTuples[1], 'r')
    trainLines = fr_train.readlines()
    testLines = fr_test.readlines()
    fr_train.close()
    fr_test.close()
    del(fr_train)
    del(fr_test)
    # save the border of train & test data for split mix-data in future
    train_test_border = len(trainLines)  
    
    '''attentional prencoder pre-processing'''
    start_org = time.clock()
    # init attentional values
    f_model, auto_attention_T = initAttentionValues(t_filepath)
    print('finish load feature selection model')
    
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    vector_dim = gensimW2VModel.vector_size
    #===========================================================================
    # load or produce gensimW2VModel, sentences, vector_seqs, attention_seqs
    # load train&test label np arrays
    #===========================================================================
    sentences = []
    vector_seqs = []
    attention_seqs = []
    labelList = []
    for line in trainLines + testLines:
        # count the parameters which needed by attentional prencoder
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        label = line[line.find(']') + 1: len(line)]
        sentences.append(words)
        lineVecs = [numpy.zeros(vector_dim)] * len(words)
        attentionVec = [0.0] * len(words)
        for i in range(len(words)):
            if words[i].decode('utf-8') in gensimW2VModel.vocab:
                lineVecs[i] = word2Vec.getWordVec(gensimW2VModel, words[i])
#                 attentionVec[i] = f_model[words[i]] if words[i] in f_model.keys() else 0.0
                attentionVec[i] = f_model[words[i]]
        vector_seqs.append(lineVecs)
        attention_seqs.append(attentionVec)
        del(attentionVec)
        
        # count the train & test labels
        classesVec = numpy.zeros(nb_classes)
        classesVec[int(label) - 1] = 1
        labelList.append(classesVec)
    del(trainLines, testLines, f_model)
    end_org = time.clock()
    print('finish load original data and generate attentional pre-data in {0}s'.format(end_org - start_org))
    
    '''run attentional prencoder'''
    start_att = time.clock()
    # switch by comment (er...)
#     attExt_vec_seqs = prencoder.seqUniDirtExt(gensimW2VModel, sentences, vector_seqs, attention_seqs, attention_T=auto_attention_T)
    attExt_vec_seqs = prencoder.seqBiDirtExt(gensimW2VModel, sentences, vector_seqs, attention_seqs, attention_T=auto_attention_T)
    del(gensimW2VModel, sentences, vector_seqs, attention_seqs)
    
    '''produce and load the attentional train & test mat data'''
    # count the max length of attext-vectors
    max_len = 0 
    for seq in attExt_vec_seqs:
        max_len = len(seq) if len(seq) > max_len else max_len
    
    # uniform the length of train & test mat data
    for i in range(len(attExt_vec_seqs)):
        attExt_vec_seqs[i].extend([numpy.zeros(vector_dim)] * (max_len - len(attExt_vec_seqs[i])))
        
    '''split train and test data, transform them into numpy array'''
    x_train = numpy.asarray(attExt_vec_seqs[:train_test_border])
    y_train = numpy.asarray(labelList[:train_test_border])
    x_test = numpy.asarray(attExt_vec_seqs[train_test_border:])
    y_test = numpy.asarray(labelList[train_test_border:])
    del(attExt_vec_seqs, labelList)
    xy_data = (x_train, y_train, x_test, y_test)
    end_att = time.clock()
    print('finish load attentional extend data and generate final exp-data in {0}s'.format(end_att - start_att))
    
    input_shape = (max_len, vector_dim)
    
    return xy_data, input_shape

def storeExprimentNpzData(npzPath, xy_data):
    start_np = time.clock()
    numpy.savez(npzPath, train_x=xy_data[0], train_y=xy_data[1], test_x=xy_data[2], test_y=xy_data[3])
    end_np = time.clock()
    print('store npz data in {0}s'.format(end_np - start_np))
    
    return npzPath

def loadExprimentNpzData(npzPath):
    start_np = time.clock()
    npzData = numpy.load(npzPath)
    xy_data = (npzData['train_x'], npzData['train_y'], npzData['test_x'], npzData['test_y'])
    input_shape = npzData['train_x'][0].shape
    end_np = time.clock()
    print('load npz data in {0}s'.format(end_np - start_np))
    
    return xy_data, input_shape
    
def trainNetworkPredictor(x_train, y_train,
                          input_shape,
                          nb_classes,
                          network='CNNs_Net',
                          frame_path=None):
    
    # reflect produce network model
    start_train = time.clock()
    model = getattr(layer, network)(input_shape, nb_classes)
    
    # record best model_weight into file
    record_path = frame_path.replace('.json', '.h5')
#     record_path = 'weights.hdf5'
    model, history_metrices = layer.trainer(model, x_train, y_train, best_record_path=record_path)
    end_train = time.clock()
    print('finish train layer model in {0}s'.format(end_train - start_train))
    print('store the best weight record on: {0}'.format(record_path))
    
    if frame_path != None:
        layer.storageModel(frame_path, replace_record=False)
        print('layer model has been stored in path: {0}.'.format(frame_path))
    
    return model, history_metrices

def loadNetworkPredictor(frame_path):
    record_path = frame_path.replace('.json', '.h5')
#     record_path = 'weights.hdf5'   
    model = layer.loadStoredModel(frame_path, record_path, recompile=True)
    print('load layer model from path: {0}.'.format(frame_path))
    return model

def runNetworkPredictor(network_model, x_test):
    
    classes, proba = layer.predictor(network_model, x_test)
    return classes, proba

def evaluateNetworkPredictor(network_model, x_test, y_test):
    
    score = layer.evaluator(network_model, x_test, y_test)
    return score

if __name__ == '__main__':
    
    '''
    test load train and test data
    '''
    trainFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/train0.txt'
    testFilePath = fileProcess.auto_config_root() + u'exp_mid_data/train_test/test0.txt'
    trainTestFileTuples = (trainFilePath, testFilePath)
    gensimW2VModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-nolabel.vector'
     
#     xy_data, input_shape = loadGensimMatData(trainTestFileTuples, gensimW2VModelPath, 11)
    xy_data, input_shape = loadAttentionGensimMatData(trainTestFileTuples, gensimW2VModelPath, 11, trainFilePath)
     
    print('x_train: {0}'.format(xy_data[0]))
    print(xy_data[0].shape)
    print('y_train: {0}'.format(xy_data[1]))
    print(xy_data[1].shape)
#     print(len(set(xy_data[1]])))
    print('x_test: {0}'.format(xy_data[2]))
    print('y_test: {0}'.format(xy_data[3]))
#     print(len(set(xy_data[3])))
    print('input_shape: {0}'.format(input_shape))

    '''
    '''
#     print(dir(layer))
#     function = getattr(layer, 'LSTM_Net')
#     model = function((100,), 10)
#     print(model)
