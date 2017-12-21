# -*- coding: UTF-8 -*-

'''
Created on 2016年11月18日

@author: superhy
'''
from interface import cacheIndex, fileProcess
from interface.embedding import word2Vec
import time


def trainWord2VecModelTest():
    
    fileProcess.reLoadEncoding()
    
    # load all file folder path
    trainDir = fileProcess.auto_config_root() + u'med_question_nolabel/'
#     med_qus_categories = cacheIndex.med_question_index.values()
#     dirPath = []
#     dirPath.extend(trainDir + category + u'/' for category in med_qus_categories)
    
#     loadedFilesPath = fileProcess.listAllFilePathInDirectory(dirPath)
#     print('files num: {0}'.format(len(loadedFilesPath)))
    
    # load all sentences to be trained
    totalSentences = fileProcess.loadMedQuesSentences(trainDir)
    print('sentences num: {0}'.format(len(totalSentences)))
    
    start_w2v = time.clock()
    w2vModelPath = fileProcess.auto_config_root() + u'model_cache/gensim/med_qus-nolabel.vector'
    model = word2Vec.trainWord2VecModel(totalSentences, w2vModelPath)
    end_w2v = time.clock()
    print('train gensim word2vec model finish, use time: {0}'.format(end_w2v - start_w2v))
    
    print('test word vector: {0}'.format(model['腰疼/v'.decode('utf-8')]))
    
    print('vocab size: {0}'.format(len(model.vocab)))
    
def loadModelfromFileTest():
    
    w2vModelPath = fileProcess.auto_config_root() + 'model_cache/gensim/med_qus-nolabel.vector'
    model = word2Vec.loadModelfromFile(w2vModelPath)
    
    queryWord = '腰疼/v'
    vector = word2Vec.getWordVec(model, queryWord)
    
    print('vector: {0}, \nvector_size: {1}'.format(vector, len(vector)))

if __name__ == '__main__':
    
    '''
    test train word2vec model
    '''
    trainWord2VecModelTest()
    
    '''
    test load word2vec model from file
    then, get a word vector
    '''
    #===========================================================================
    # loadModelfromFileTest()
    #===========================================================================