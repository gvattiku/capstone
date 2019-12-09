# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:56:16 2019

@author: amolv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
import os

os.chdir('D:/Saumya/DAEN690')
os.getcwd()
df = pd.read_excel("SyntheticData-XgBoost model-Coursegrade.xlsx")
df.info()
mean = df['Course.Grade.with.No.extra.credit'].mean()
mean
df['Course.Grade.with.No.extra.credit'].fillna(mean, inplace=True)
df=df.drop(['Course.Grade.with.Extra.Credit',
            'Proj','HW10','HW9','HW8','HW7','HW6','SectionCode','Term',
            'Season','Year','Type','GTA','Major','Domicile','Credit hours',
            'Test2','Finals','PreReqSatisfied','Class','Test1','Instructor'],axis=1)
df.info()

#one hot encoding
dfcat=df.drop(['HW1','HW2','HW3','HW4','HW5','Quizzes',
               'Course.Grade.with.No.extra.credit'],axis=1)
dfcat.info()
one_hot = pd.get_dummies(dfcat)
one_hot.info()
dfnum=df.drop(['Level','Load','gender'],axis=1)
dffinal= dfnum.join(one_hot)
dffinal.info()
dffinal.head() 

#Splitting data to test and train set 
X =dffinal.drop('Course.Grade.with.No.extra.credit',axis=1) 
Y=dffinal['Course.Grade.with.No.extra.credit']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#Model
from sklearn.svm import SVR

model = SVR(kernel='linear')
model.fit(X_train,Y_train)


#predicting the response
ypred = model.predict(X_test)
ypred

#Results
mse = mean_squared_error(Y_test,ypred)
print("MSE: %.2f" % mse)

errors = abs(ypred - Y_test)
print('Mean Absolute Error:', round(np.mean(errors),2), '')
rmse = np.sqrt(mean_squared_error(Y_test, ypred))
print("RMSE: %f" % (rmse))
print(model.score(X, Y), 1 - (1-model.score(X,Y))*(len(Y)-1)/(len(Y)-X.shape[1]-1))


from matplotlib import pyplot
from xgboost import plot_importance
plot_importance(model)
pyplot.show()
