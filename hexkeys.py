#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import hashlib, binascii
import os

"""
Set a password
"""
password = "trekEmondaSlr9D"

"""
Set the path for the work directory
"""
#os.chdir('/home/r00t/Work/capstone')

"""
Generate Hashes
"""
def hashKey(data):
    global password
    keyGen = hashlib.pbkdf2_hmac('md5', data.encode('utf-8'), password.encode('utf-8'), 100000)
    return binascii.hexlify(keyGen)

"""
Set the root directory
"""
root = '/home/r00t/Work/capstone/Data/Input'

"""
Work on the all files in directory
"""
for file in os.listdir(root):
    df = pd.read_csv(root + '/' + file)
    df['Key'] = df['Student ID'].apply(hashKey)
    
    dfKeyMap = df[['Key', 'Student ID']]
    dfKeyMap.drop_duplicates(subset ='Student ID', keep ='last')
    
    df = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis =1)
    df.to_csv('/home/r00t/Work/capstone/Data/Output/' +file + "_anonim_data.csv")	
    
dfKeyMap.to_csv('/home/r00t/Work/capstone/Data/Keys/'+ 'studentKeys.csv', index = False)

"""
def hashMd5Key(data):
    y = md5()
    y.update(data.encode("utf-8"))
    return y.hexdigest()

col = ['Hex','Student ID']
keyDf = pd.DataFrame(columns = col)
#keyDf

import glob
fileList = glob.glob(r'C:\Users\vatti\Downloads\Capstone')
fileList

import os
root = r'C:\Users\vatti\Downloads\Capstone'
for file in os.listdir(root):
    df=pd.read_csv(root+'/'+file)
    df['Student ID']

df['Student ID'].apply(hashMd5Key)

for item in fileList:
    file = pd.read_csv(item)
    file['Hex'] = file['Student ID'].apply(hashKey)
    keyDf[['Hex', 'Student ID']] = file[['Hex', 'Student ID']]
    keyDf.drop_duplicates(subset='Student ID', keep = last)

keyDf.to_csv(r'C:\Users\vatti\Downloads\Capstone\keyFile.csv', sep=',', encoding='utf-8')

finalDf = file.drop(file.columns[[0, 1, 2, 3, 4, 5]], axis=1) 
finalDf.to_csv(r'C:\Users\vatti\Downloads\Capstone\anonym.csv', sep=',', encoding='utf-8')

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