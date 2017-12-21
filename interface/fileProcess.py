# -*- coding: UTF-8 -*-

'''
Created on 2016年11月17日

@author: superhy
'''

from gensim.models import word2vec
import os
import platform
import types
import warnings

import cacheIndex


root_windows = 'D:\\intent-rec-file\\'
root_macos = ''
root_linux = '/intent-rec-file/'

root_server225 = '/Code/experiment/huyang/data/intent-rec-file/'

if platform.system() == 'Linux':
    root_linux = os.environ['HOME'] + root_linux
    # server TITAN-X 225 root_path
    root_server225 = os.environ['HOME'] + root_server225

def auto_config_root():

    global root_linux
    global root_macos
    global root_windows
    
    global root_server225

    if platform.system() == 'Windows':
        return root_windows
    elif platform.system() == 'Linux':
        if platform.uname()[1] == 'server225':
            return root_server225
        else:
            return root_linux
    else:
        return None

def reLoadEncoding():
        # 重新载入字符集
        import sys
        reload(sys)
        sys.setdefaultencoding('utf-8')
        
def listAllFilePathInDirectory(dirPath):
    '''
    list all file_path in a directory from dir folder
    '''
    reLoadEncoding()

    loadedFilesPath = []
    if type(dirPath) is types.StringType or type(dirPath) is types.UnicodeType:
        # dirPath is just a string
        files = os.listdir(dirPath)
        loadedFilesPath.extend(dirPath + file for file in files)
    elif type(dirPath) is types.ListType:
        # dirPath is a list which own many dir's paths
        for dir in dirPath:
            part_files = []
            part_files.extend(os.listdir(dir))
            loadedFilesPath.extend(dir + file for file in part_files)
    else:
        loadedFilesPath = None        
        warnings.warn('input dirPath type is wrong!')
    
    return loadedFilesPath

def loadMedQuesSentences(totalDataPath):
#     totalDataPath = fileProcess.auto_config_root() + 'med_question_5000each/'

    reLoadEncoding()

    # list all file data path
    med_qus_categories = cacheIndex.med_question_index.keys()
    dirPath = []
    dirPath.extend(totalDataPath + category + '/' for category in med_qus_categories)
    print('load index:')
    for dir in dirPath:
        print(dir)
    loadedFilesPath = listAllFilePathInDirectory(dirPath)
    
    totalSentences = []
    for filePath in loadedFilesPath:
        totalSentences.extend(word2vec.LineSentence(filePath))
    
    # loaded text encode is 'utf-8'
    return totalSentences

if __name__ == '__main__':
    
    trainDir = auto_config_root() + 'med_question_5000each/'
    med_qus_categories = cacheIndex.med_question_index.keys()
    dirPath = []
    dirPath.extend(trainDir + category + '/' for category in med_qus_categories)
    
    loadedFilesPath = listAllFilePathInDirectory(dirPath)
#     for file in loadedFilesPath:
#         print(file)
    print('files num: ' + str(len(loadedFilesPath)))
    
    totalSentences = loadMedQuesSentences(trainDir)
    print('sentences num: ' + str(len(totalSentences)))
    print(type(totalSentences[0][0].encode('utf-8')))
