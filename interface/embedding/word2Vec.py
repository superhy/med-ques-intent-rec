# -*- coding: UTF-8 -*-

'''
Created on 2016年11月17日

@author: superhy
'''
import multiprocessing
from gensim.models.word2vec import Word2Vec
from interface import fileProcess

def trainWord2VecModel(sentences, modelPath,
                           Size=100,
                           Window=5,
                           MinCount=1,
                           Workers=multiprocessing.cpu_count()):
    # reload for safe
    fileProcess.reLoadEncoding()
    
    # train word2vec model
    model = Word2Vec(sentences, 
                     size=Size, window=Window, min_count=MinCount, workers=Workers)
    #===========================================================================
    # save work2vec model on disk
    # then, load sim_data
    #===========================================================================
    model.save(modelPath)
    model.init_sims(replace=False)
    print('producing word2vec model ... ok! model store in {0}'.format(modelPath))

    return model

def getWordVec(model, queryWord):
    # reload for safe
    fileProcess.reLoadEncoding()
    
    vector = model[queryWord.decode('utf-8')]
#     vector = model[queryWord]
    return vector

def queryMostSimWords(model, wordStr, topN=20):
    '''
    MSimilar words basic query function
    return 2-dim List [0] is word [1] is double-prob
    '''
    fileProcess.reLoadEncoding()
        
    similarPairList = model.most_similar(wordStr.decode('utf-8'), topn=topN)
    return similarPairList

def loadModelfromFile(modelPath):
    
    model = Word2Vec.load(modelPath)
    return model

if __name__ == '__main__':
    pass
