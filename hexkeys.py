#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from hashlib import md5

def hashMd5Key(data):
    y = md5()
    y.update(data.encode("utf-8"))
    return y.hexdigest()


# In[14]:


import glob
fileList = glob.glob('')
fileList

for item in fileList:
    file = pd.read_csv(item)
    file['Hex'] = file['Student ID'].apply(hashMd5Key)
    file[['Student ID', 'Hex']].to_csv('', sep=',', encoding='utf-8')


# In[9]:


df = file.drop(file.columns[[0, 1, 2, 3, 4, 5]], axis=1) 
df.to_csv('', sep=',', encoding='utf-8')

