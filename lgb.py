import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd  
import random
import seaborn as snb


df = pd.read_csv("C:\\Users\\vidhy\\OneDrive\\Desktop\\Fall 19\\DEAN 690\\UI\\SyntheticData.csv")
df=df.drop(columns=['Num'])
mean = df['Course.Grade.with.No.extra.credit'].mean()
df['Course.Grade.with.No.extra.credit'].fillna(mean, inplace=True)
df=df.drop(['Course.Grade.with.Extra.Credit',
            'Proj','HW10','HW9','HW8','HW7','HW6','SectionCode','Term',
            'Season','Year','Type','GTA','Domicile','Credit hours',
            'Test2','Finals','PreReqSatisfied','Class','Test1','Instructor'],axis=1)

#One-hot encoding for converting categorical to dichotomous variables
dfcat=df.drop(['HW1','HW2','HW3','HW4','HW5','Quizzes','Course.Grade.with.No.extra.credit'],axis=1)
one_hot = pd.get_dummies(dfcat)
dfnum=df.drop(['Level','Load','gender','Major'],axis=1)
dffinal= dfnum.join(one_hot)

#Split train/test
X = dffinal.drop('Course.Grade.with.No.extra.credit', axis=1)
y = dffinal['Course.Grade.with.No.extra.credit']


random.seed(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


#lightgbm boost model
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'huber',
    'metric': 'rmse',
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 10,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 500000,
    "n_estimators": 1000,
    'random_state':0,
    'silent':1
}

gbm = lgb.LGBMRegressor(**hyper_params)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        verbose=True,   
        early_stopping_rounds=1000)

y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)

print('The rmse of prediction is:', round(mean_squared_error(y_pred, y_train) ** 0.5, 5))

#feature importance plot
feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importances_,X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 5))
snb.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

#store results to csv
test_pred=np.expm1(gbm.predict(X_test))
X_test["Course.Grade.with.No.extra.credit"] = np.log1p(test_pred)
X_test.to_csv("lgbresults.csv", columns=["Course.Grade.with.No.extra.credit"], index=False)
