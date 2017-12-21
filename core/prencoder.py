# -*- coding: UTF-8 -*-

'''
Created on 2016年12月2日

@author: superhy
'''
import warnings
import numpy
from interface import word2Vec
import random
from numpy import dtype

def loadAttSimVecDic(gensimW2VModel, sentences, attention_seqs, attention_T):
    if len(sentences) != len(attention_seqs):
        warnings.warn('given incompatible dim_sentences!')
        
    attSimVecDic = {}
    for i in range(len(sentences)):
        for j in range(len(attention_seqs[i])):
            if attention_seqs[i][j] >= attention_T:
                if sentences[i][j] not in attSimVecDic.keys() and sentences[i][j].decode('utf-8') in gensimW2VModel.vocab:
                    simWord = word2Vec.queryMostSimWords(gensimW2VModel, sentences[i][j], topN=1)[0][0]
                    simVec = word2Vec.getWordVec(gensimW2VModel, simWord)
                    attSimVecDic[sentences[i][j]] = simVec
                    
    return attSimVecDic

def genExtVecs(attVec, simVec, tagVecs, extNum):
    '''
    @param @tagVecs: tuple: (1, ) in seqUniDirtExt or (1, 2) which right-1 left-2 in seqBiDirtExt
    '''
    N_1 = 1.0 / 8
    ''' numpy array calculation '''
    varyRange = (attVec - simVec) * N_1
    # vary from big to small
    varyDecay = varyRange * 1.0 / extNum
    extVecs = []
    
    def addCopyExtVecs(tagIndex, j):
        extVecs.append(attVec)
    
    def addRadomSkipExtVecs(tagIndex, j):
        random_v = (random.randint(20, 60) * 1.0 / 100) * varyDecay
        varyVec = varyRange - varyDecay * (j - 1) - random_v
        varydirtVec = numpy.asarray(list(1 if tagVecs[tagIndex][i] > attVec[i] else -1 for i in range(len(attVec))))
        extVecs.append(attVec + varyVec * varydirtVec)
        
    def addExpDecayExtVecs(tagIndex, j):
        varyVec = varyRange * (1.0 - numpy.e ** (-1.0 * (extNum - j) / (extNum)))
        varydirtVec = numpy.asarray(list(1 if tagVecs[tagIndex][i] > attVec[i] else -1 for i in range(len(attVec))))
        extVecs.append(attVec + varyVec * varydirtVec)
        
    if len(tagVecs) == 1: 
        for j in range(extNum):
#             addCopyExtVecs(0, j)
#             addRadomSkipExtVecs(0, j)
            addExpDecayExtVecs(0, j)
    elif len(tagVecs) == 2:
        for j in range(extNum):
#             addCopyExtVecs(j % 2, j)
#             addRadomSkipExtVecs(j % 2, j)
            addExpDecayExtVecs(j % 2, j)
    else:
        warnings.warn('nb_tagVecs exceed the limit! use first-2')
        for j in range(extNum):
#             addCopyExtVecs(j % 2, j)
#             addRadomSkipExtVecs(j % 2, j)
            addExpDecayExtVecs(j % 2, j)
            
    return extVecs

def seqUniDirtExt(gensimW2VModel, sentences, vector_seqs, attention_seqs, attention_T, ext_lemda=(0.25, 1.0)):
    '''
    @param @ext_lemda: tuple contain 2 elements, 0: baseline_lemda to calculation baseline ext-length,
        1: limit_lemda to calculation max ext-length
    '''
    
    len_vectorSeqs = len(vector_seqs)
    len_attentionSeqs = len(attention_seqs)
    if len_attentionSeqs != len_vectorSeqs:
        warnings.warn('given incompatible dim_sequences!')
        
    # load all attSimVecDic firstly
    attSimVecDic = loadAttSimVecDic(gensimW2VModel, sentences, attention_seqs, attention_T)
    del(gensimW2VModel)  # release the memory space
    
    # count the average value of vector sequence's length
    avelen_vecSeq = numpy.mean(list(len(vecSeq) for vecSeq in vector_seqs))
    extNum_b = ext_lemda[0] * avelen_vecSeq
    extNum_l = int(ext_lemda[1] * avelen_vecSeq)
    
    attExt_vec_seqs = []
    # count some attention extend info
    ave_nb_extNum = 0.0
    ave_nb_extIndexes = 0.0
    ave_extLen = 0.0
    ave_len_attVec = 0.0
    for i in range(len_vectorSeqs):
        # count the extension range from extension length base
        extNum = int(round(extNum_b * avelen_vecSeq * 1.0 / len(vector_seqs[i]))) if extNum_b * avelen_vecSeq * 1.0 / len(vector_seqs[i]) > 1 else 1
        extNum = extNum if extNum <= extNum_l else extNum_l
        
        # record the elements' indexes which need extension
        extIndexes = []
        extAttValue = []  # store the attention value of each element to control the extNum, index same as extIndexes
        for att_i in range(len(attention_seqs[i])):
#             print(att_i)
            if attention_seqs[i][att_i] >= attention_T:
                extIndexes.append(att_i)
                extAttValue.append(attention_seqs[i][att_i])
        extIndexes = numpy.asarray(extIndexes)
#         for i in range(len(extIndexes)):
#             print('{0} '.format(extIndexes[i])),
#         print('')
        
        org_vec_seq = vector_seqs[i]
        
        # count some statics info (1)
        ave_nb_extNum += (extNum * 1.0 / len_vectorSeqs)  # basic expand number
        ave_nb_extIndexes += (len(extIndexes) * 1.0 / len_vectorSeqs)  # number of expand elements
        
        # doing the extension
        push = 0
        for j in range(len(extIndexes)):
#             print('len sentence: {0}, len attention seq: {1}, extIndex: {2}'.format(len(sentences[i]), len(attention_seqs[i]), extIndexes[j]))
            if extIndexes[j] + push == len(org_vec_seq) - 1:
                # check index on right border, only extension vectors on left direction
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    continue
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] - 1],)
#                 extVecs = genExtVecs(attVec, simVec, tagVecs, (int(extNum * extAttValue[j]) + 1) / 2)
                extVecs = genExtVecs(attVec, simVec, tagVecs, (extNum + 1) / 2)
                del(attVec, simVec, tagVecs)  # release the memory space
                
                for ext_i in range(len(extVecs)):
                    org_vec_seq.insert(extIndexes[j] + push + ext_i, extVecs[ext_i])  # after insert on left, att_ele always be pushed one step
                # push the rest indexs
                push += len(extVecs)
                del(extVecs)  # release the memory space
            else:
                # normally only extension vectors on right directions
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    continue
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] + 1],)
#                 extVecs = genExtVecs(attVec, simVec, tagVecs, int(extNum * extAttValue[j]))
                extVecs = genExtVecs(attVec, simVec, tagVecs, extNum)
                del(attVec, simVec, tagVecs)  # release the memory space
                
                for ext_i in range(len(extVecs)):
                    # when insert on left, att_ele has been pushed one step
                    # so we need carefully about this, push forward the insert position
                    # the more decay, the further from the original vector
                    org_vec_seq.insert(extIndexes[j] + push + 1, extVecs[ext_i])
                # push the rest indexs
                push += len(extVecs)
                del(extVecs)  # release the memory space
        
        attExt_vec_seqs.append(org_vec_seq)
        
        # count some statics info(2)
        ave_extLen += (push * 1.0 / len_vectorSeqs)
        ave_len_attVec += (len(org_vec_seq) * 1.0 / len_vectorSeqs)
        
        # release the memory space
        del(org_vec_seq, extIndexes, extAttValue, extNum, push)
    
    print('average expand length base: {0}'.format(ave_nb_extNum))
    print('average number of expand elements: {0}'.format(ave_nb_extIndexes))
    print('average expand length: {0}'.format(ave_extLen))
    print('average length of expanded vectors: {0}'.format(ave_len_attVec))
    
    return attExt_vec_seqs

def seqBiDirtExt(gensimW2VModel, sentences, vector_seqs, attention_seqs, attention_T, ext_lemda=(0.25, 1.0)):
    '''
    @param @ext_lemda: tuple contain 2 elements, 0: baseline_lemda to calculation baseline ext-length,
        1: limit_lemda to calculation max ext-length
    '''
    
    len_vectorSeqs = len(vector_seqs)
    len_attentionSeqs = len(attention_seqs)
    if len_attentionSeqs != len_vectorSeqs:
        warnings.warn('given incompatible dim_sequences!')
        
    # load all attSimVecDic firstly
    attSimVecDic = loadAttSimVecDic(gensimW2VModel, sentences, attention_seqs, attention_T)
    del(gensimW2VModel)  # release the memory space
    
    # count the average value of vector sequence's length
    avelen_vecSeq = numpy.mean(list(len(vecSeq) for vecSeq in vector_seqs))
    extNum_b = ext_lemda[0] * avelen_vecSeq
    extNum_l = int(ext_lemda[1] * avelen_vecSeq)
    
    attExt_vec_seqs = []
    # count some attention extend info
    ave_nb_extNum = 0.0
    ave_nb_extIndexes = 0.0
    ave_extLen = 0.0
    ave_len_attVec = 0.0
    for i in range(len_vectorSeqs):
        # count the extension range from extension length base
        extNum = int(round(extNum_b * avelen_vecSeq * 1.0 / len(vector_seqs[i]))) if extNum_b * avelen_vecSeq * 1.0 / len(vector_seqs[i]) > 1 else 1
        extNum = extNum if extNum <= extNum_l else extNum_l
        
        # record the elements' indexes which need extension
        extIndexes = []
        extAttValue = []  # store the attention value of each element to control the extNum, index same as extIndexes
        for att_i in range(len(attention_seqs[i])):
#             print(att_i)
            if attention_seqs[i][att_i] >= attention_T:
                extIndexes.append(att_i)
                extAttValue.append(attention_seqs[i][att_i])
        extIndexes = numpy.asarray(extIndexes)
#         for i in range(len(extIndexes)):
#             print('{0} '.format(extIndexes[i])),
#         print('')
        
        org_vec_seq = vector_seqs[i]
        
        # count some statics info (1)
        ave_nb_extNum += (extNum * 1.0 / len_vectorSeqs)  # basic expand number
        ave_nb_extIndexes += (len(extIndexes) * 1.0 / len_vectorSeqs)  # number of expand elements
        
        # doing the extension
        push = 0
        for j in range(len(extIndexes)):
#             print('len sentence: {0}, len attention seq: {1}, extIndex: {2}'.format(len(sentences[i]), len(attention_seqs[i]), extIndexes[j]))
            if extIndexes[j] + push == 0:
                # check index on left border, only extension half vectors on right direction
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    continue
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] + 1],)
#                 extVecs = genExtVecs(attVec, simVec, tagVecs, (int(extNum * extAttValue[j]) + 1) / 2)
                extVecs = genExtVecs(attVec, simVec, tagVecs, (extNum + 1) / 2)
                del(attVec, simVec, tagVecs)  # release the memory space
                
                for ext_i in range(len(extVecs)):
                    org_vec_seq.insert(extIndexes[j] + push + 1, extVecs[ext_i])
                # push the rest indexs
                push += len(extVecs)
                del(extVecs)  # release the memory space
            elif extIndexes[j] + push == len(org_vec_seq) - 1:
                # check index on right border, only extension half vectors on left direction
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    continue
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] - 1],)
#                 extVecs = genExtVecs(attVec, simVec, tagVecs, (int(extNum * extAttValue[j]) + 1) / 2)
                extVecs = genExtVecs(attVec, simVec, tagVecs, (extNum + 1) / 2)
                del(attVec, simVec, tagVecs)  # release the memory space
                
                for ext_i in range(len(extVecs)):
                    org_vec_seq.insert(extIndexes[j] + push + ext_i, extVecs[ext_i])  # after insert on left, att_ele always be pushed one step
                # push the rest indexs
                push += len(extVecs)
                del(extVecs)  # release the memory space
            else:
                # extension vectors on both right & left directions
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    continue
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] + 1],)
#                 extVecs = genExtVecs(attVec, simVec, tagVecs, int(extNum * extAttValue[j]))
                extVecs = genExtVecs(attVec, simVec, tagVecs, extNum)
                del(attVec, simVec, tagVecs)  # release the memory space
                
                for ext_i in range(len(extVecs)):
                    # when insert on left, att_ele has been pushed one step
                    # so we need carefully about this, push forward the insert position
                    # the more decay, the further from the original vector
                    org_vec_seq.insert(extIndexes[j] + push + ext_i / 2 + (ext_i + 1) % 2, extVecs[ext_i])
                # push the rest indexs
                push += len(extVecs)
                del(extVecs)  # release the memory space
        
        attExt_vec_seqs.append(org_vec_seq)
        
        # count some statics info(2)
        ave_extLen += (push * 1.0 / len_vectorSeqs)
        ave_len_attVec += (len(org_vec_seq) * 1.0 / len_vectorSeqs)
        
        # release the memory space
        del(org_vec_seq, extIndexes, extAttValue, extNum, push)
    
    print('average expand length base: {0}'.format(ave_nb_extNum))
    print('average number of expand elements: {0}'.format(ave_nb_extIndexes))
    print('average expand length: {0}'.format(ave_extLen))
    print('average length of expanded vectors: {0}'.format(ave_len_attVec))
    
    return attExt_vec_seqs
    
if __name__ == '__main__':
    pass
