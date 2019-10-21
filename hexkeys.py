#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
from hashlib import md5

def hashMd5Key(data):
    y = md5()
    y.update(data.encode("utf-8"))
    return y.hexdigest()


# In[27]:


"""
col = ['Hex','Student ID']
keyDf = pd.DataFrame(columns = col)
#keyDf

import glob
fileList = glob.glob(r'C:\Users\vatti\Downloads\Capstone')
fileList
"""


# In[34]:


import os
root = r'C:\Users\vatti\Downloads\Capstone'
for file in os.listdir(root):
    df=pd.read_csv(root+'/'+file)
    df['Student']


# In[60]:


import hashlib, binascii
password_to_prof = "afasf"
def hashKey(data):
    global password_to_prof
    dk = hashlib.pbkdf2_hmac('md5', data.encode('utf-8'), password_to_prof.encode('utf-8'), 100000)
    return binascii.hexlify(dk)


# In[69]:


import os
root = r'C:\Users\vatti\Downloads\Capstone'
for file in os.listdir(root):
    df=pd.read_csv(root+'/'+file)
    df['key'] = df['Student ID'].apply(hashKey)
    df_key_mapping = df[['key', 'Student ID']]
    # drop duplicates from df_new.
    df_key_mapping.drop_duplicates(subset='Student ID', keep = last)
    df_key_mapping.to_csv(file+'_keys.csv')
    df = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1)
    df.to_csv(file+"_anonim_data.csv")
    


# In[42]:


#df['Student ID'].apply(hashMd5Key)


# In[61]:


"""
for item in fileList:
    file = pd.read_csv(item)
    file['Hex'] = file['Student ID'].apply(hashKey)
    keyDf[['Hex', 'Student ID']] = file[['Hex', 'Student ID']]
    keyDf.drop_duplicates(subset='Student ID', keep = last)
"""


# In[ ]:


"""
keyDf.to_csv(r'C:\Users\vatti\Downloads\Capstone\keyFile.csv', sep=',', encoding='utf-8')

finalDf = file.drop(file.columns[[0, 1, 2, 3, 4, 5]], axis=1) 
finalDf.to_csv(r'C:\Users\vatti\Downloads\Capstone\anonym.csv', sep=',', encoding='utf-8')
"""


# In[ ]:


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
        
    if file['Student ID'] not in keyDf['Studend ID']:
        file['Hex'] = file['Student ID'].apply(hashMd5Key)
        keyDf[[0, 1]] = file[['Hex', 'Student ID']]
"""

