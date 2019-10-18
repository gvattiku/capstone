#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from hashlib import md5

def hashMd5Key(data):
    y = md5(key='CanNondALe')
    y.update(data.encode("utf-8"))
    return y.hexdigest()


# In[22]:


col = ['Hex','Student ID']
keyDf = pd.DataFrame(columns = col)
#keyDf

import glob
fileList = glob.glob('')
#fileList

"""
def check(list1, list2):
    for item in list1:
        if item in list2:
            return False
        else:
            return True

for item in fileList:
    file = pd.read_csv(item)
    while check(keyDf['Student ID'], file['Student ID']):
        file['Hex'] = file['Student ID'].apply(hashMd5Key)
        keyDf[['Hex', 'Student ID']] = file[['Hex', 'Student ID']]
"""       

for item in fileList:
    file = pd.read_csv(item)
    file['Hex'] = file['Student ID'].apply(hashMd5Key)
    keyDf[['Hex', 'Student ID']] = file[['Hex', 'Student ID']]
    keyDf.drop_duplicates(subset='Student ID', keep = last)

"""
    if file['Student ID'] not in keyDf['Studend ID']:
        file['Hex'] = file['Student ID'].apply(hashMd5Key)
        keyDf[[0, 1]] = file[['Hex', 'Student ID']]
    else:
"""

keyDf.to_csv('', sep=',', encoding='utf-8')

finalDf = file.drop(file.columns[[0, 1, 2, 3, 4, 5]], axis=1) 
finalDf.to_csv('', sep=',', encoding='utf-8')

