# -*- coding: UTF-8 -*-

'''
Created on 2016年11月20日

@author: super
'''

import random
import time
import warnings

from interface import  fileProcess, cacheIndex
from interface.embedding import word2Vec


# _totalDirPath_1 = fileProcess.auto_config_root() + u'med_question_1000each/'
_totalDirPath_2_5 = fileProcess.auto_config_root() + u'med_question_2500each/'
_totalDirPath_3_5 = fileProcess.auto_config_root() + u'med_question_3500each/'
_totalDirPath_5 = fileProcess.auto_config_root() + u'med_question_5000each/'

def prodRandomLabeledData(totalDirPath, writeFilePath_5=None):
    
    fileProcess.reLoadEncoding()
    
    # load all sentences to be trained
    totalSentences = fileProcess.loadMedQuesSentences(totalDirPath)
    
    med_qus_categories = cacheIndex.med_question_index.keys()
#     dirPath = []
#     dirPath.extend(totalDirPath + category + '/' for category in med_qus_categories)
    start_label = time.clock()
    classes = []
    for category in med_qus_categories:
        cateDirPath = totalDirPath + category + '/'
        cateFilesPath = fileProcess.listAllFilePathInDirectory(cateDirPath)
        for i in range(len(cateFilesPath)):
            classes.append(cacheIndex.med_question_index[category])
            
    totalSentences_labeled = []
    for i in range(len(totalSentences)):
        words_str = '[' + ','.join(totalSentences[i]) + ']'
        sentence_labeled = words_str + str(classes[i])
        totalSentences_labeled.append(sentence_labeled)
    end_label = time.clock()
    print('finish give labels in {0}s'.format(end_label - start_label))
    
    start_random = time.clock()
    random.shuffle(totalSentences_labeled)
    end_random = time.clock()
    print('finish random data in {0}s'.format(end_random - start_random))
    
    if writeFilePath_5 != None:
        fw = open(writeFilePath_5, 'w')
        fw.write('\n'.join(totalSentences_labeled))
        fw.close()
    
    return totalSentences_labeled

def splitTrainTestData(totalDataPath, trainTestDirPath, split_rate=10):
    '''
    @param totalDataPath: string of data path which has all sentence line with labels
    @param split_rate: positive integer which means number of split pieces, default is 10
    '''
    
    fileProcess.reLoadEncoding()
    
    fw = open(totalDataPath, 'r')
    totalLines = fw.readlines()
    fw.close()
    
    start_split = time.clock()
    nb_lines = len(totalLines)
    if nb_lines % split_rate != 0:
        warnings.warn('split_rate must can divide number of lines!')
        return None
    
    span = nb_lines / split_rate
#     splitPartLines = []
    splitTrainTestTuples = []
    scan_p = 0
    for i in range(split_rate):
#         splitPartLines.append(totalLines[scan_p : scan_p + span])
        print('split from ' + str(0) + ' to ' + str(scan_p) + ', ' + str(scan_p + span) + ' to ' + str(nb_lines))
        splitTrainTestTuples.append((totalLines[0:scan_p] + totalLines[scan_p + span:nb_lines], totalLines[scan_p:scan_p + span]))
        scan_p += span
    end_split = time.clock()
    print('finish split train and test data in {0}s'.format(end_split - start_split))
    
    start_write = time.clock()
    for i in range(len(splitTrainTestTuples)):
        writeTrainTestPathTuple = (trainTestDirPath + u'train{0}.txt'.format(i), trainTestDirPath + u'test{0}.txt'.format(i))
        writeTrainStr = ''.join(splitTrainTestTuples[i][0])
        writeTestStr = ''.join(splitTrainTestTuples[i][1])
        
        fw_train = open(writeTrainTestPathTuple[0], 'w')
        fw_train.write(writeTrainStr)
        fw_train.close()
        fw_test = open(writeTrainTestPathTuple[1], 'w')
        fw_test.write(writeTestStr)
        fw_test.close()
    end_write = time.clock()
    print('finish produce split train test data file in {0}s'.format(end_write - start_write))
    
    return splitTrainTestTuples

def statTextInfo(totalDataPath):
    fr = open(totalDataPath, 'r')
    lines = fr.readlines()
    fr.close()
    
    ave = 0.0
    min = 0xFFFFFF * 1.0
    max = 0.0
    for line in lines:
        words = line[line.find('[') + 1 : line.find(']')].split(',')
        ave += (len(words) * 1.0 / len(lines))
        min = len(words) * 1.0 if min > len(words) * 1.0 else min
        max = len(words) * 1.0 if max < len(words) * 1.0 else max
    
    return ave, min, max

if __name__ == '__main__':
    
    writeFilePath_2_5 = fileProcess.auto_config_root() + u'exp_mid_data/sentences_labeled27500.txt'
    writeFilePath_3_5 = fileProcess.auto_config_root() + u'exp_mid_data/sentences_labeled38500.txt'
    writeFilePath_5 = fileProcess.auto_config_root() + u'exp_mid_data/sentences_labeled55000.txt'
    
#     prodRandomLabeledData(_totalDirPath_2_5, writeFilePath_2_5)
    prodRandomLabeledData(_totalDirPath_3_5, writeFilePath_3_5)
#     prodRandomLabeledData(_totalDirPath_5, writeFilePath_5)
    
    '''
    test mid data index in gensim word2vec
    '''
    #===========================================================================
    # fr = open(writeFilePath_5, 'r')
    # line = fr.readline()
    # print(type(line))
    # test_words = line[line.find('[') + 1:line.find(']')].split(',')
    # print(test_words[len(test_words) - 1])
    #    
    # w2vModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-nolabel.vector'
    # model = word2Vec.loadModelfromFile(w2vModelPath)
    #    
    # vector = word2Vec.getWordVec(model, test_words[len(test_words) - 1])
    # print(type(vector))
    #===========================================================================
    
    '''
    '''
    trainTestDir = fileProcess.auto_config_root() + u'exp_mid_data/train_test-3500/'
    splitTrainTestData(writeFilePath_3_5, trainTestDir)
    
    '''
    '''
    ave, min, max = statTextInfo(writeFilePath_3_5)
    print('ave: {0}, min: {1}, max: {2}'.format(ave, min, max))
