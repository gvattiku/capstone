
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd 
from matplotlib import pyplot 
from sklearn import metrics
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
            'Season','Year','Type','GTA','Domicile','Credit hours',
            'Test2','Finals','PreReqSatisfied','Class','Test1','Instructor'],axis=1)
df.info()

#One hot encoding
dfcat=df.drop(['HW1','HW2','HW3','HW4','HW5','Quizzes',
               'Course.Grade.with.No.extra.credit'],axis=1)
dfcat.info()
one_hot = pd.get_dummies(dfcat)
one_hot.info()
dfnum=df.drop(['Level','Load','gender','Major'],axis=1)
dffinal= dfnum.join(one_hot)
dffinal.info()
 
#Split to train and test
X = dffinal.drop('Course.Grade.with.No.extra.credit', axis=1)
y = dffinal['Course.Grade.with.No.extra.credit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)

#Model
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
preds
  
r_sq = model.score(X, y)
print('coefficient of determination:', r_sq)
print(metrics.mean_absolute_error(y_test,preds))
print(metrics.mean_squared_error(y_test,preds))
print(np.sqrt(metrics.mean_squared_error(y_test,preds)))



    

