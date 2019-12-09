import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd  
import random

#read the data and fill the missing values with mean for the target variable
df = pd.read_csv("C:\\Users\\vidhy\\OneDrive\\Desktop\\Fall 19\\DEAN690\\UI\\SyntheticData.csv")
df=df.drop(columns=['Num'])
mean = df['Course.Grade.with.No.extra.credit'].mean()

#One hot encoding
df['Course.Grade.with.No.extra.credit'].fillna(mean, inplace=True)
df=df.drop(['Course.Grade.with.Extra.Credit', 
            'Proj','HW10','HW9','HW8','HW7','HW6','SectionCode','Term',
            'Season', 'Year', 'Type', 'GTA', 'Domicile', 'Credit hours',
            'Test2','Finals','PreReqSatisfied','Class','Test1','Instructor'], axis=1)
dfcat=df.drop(['HW1','HW2','HW3','HW4','HW5','Quizzes','Course.Grade.with.No.extra.credit'],axis=1)
one_hot = pd.get_dummies(dfcat)
dfnum=df.drop(['Level','Load','gender','Major'],axis=1)
dffinal= dfnum.join(one_hot)

#Split train/test
X = dffinal.drop('Course.Grade.with.No.extra.credit', axis=1)
y = dffinal['Course.Grade.with.No.extra.credit']

#xgbregressor
random.seed(1111)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)

model = xgb.XGBRegressor(random_state=1111)
model.fit(X_train,y_train)


#Grid seacrh

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
          	'learning_rate': [.01,.03, 0.05,0.07,0.09], #so called `eta` value
          	'max_depth': [1,2,3,4,5],
          	#'gamma': [i/10.0 for i in range(0,5)],
          	'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
          	'min_child_weight': [1,4,8,10],
          	'silent': [1],
          	'subsample': [0.6,0.7,0.8,1.0],
          	'colsample_bytree': [0.6,0.7,0.8,1.0],
          	'n_estimators': [1000]}

xgb_grid = GridSearchCV(model,
                    	parameters,
                    	cv = 2,
                    	n_jobs = 5,
                    	verbose=True,
                    	scoring='neg_mean_squared_error')

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)  
print(xgb_grid.best_estimator_)  

model=xgb.XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=1,
   	colsample_bynode=1, colsample_bytree=1, gamma=0,
   	importance_type='gain', learning_rate=0.1, max_delta_step=0,
   	max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
   	n_jobs=1, nthread=None, random_state=1111,
   	reg_alpha=100, reg_lambda=1, scale_pos_weight=1, seed=None,
   	silent=1, subsample=1, verbosity=1)

model.fit(X_train,y_train)
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model,X_train,y_train, cv=kfold, scoring='neg_mean_squared_error' )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

#predict test data and check its accuracy. We'll use MSE and RMSE as accuracy metrics.
ypred = model.predict(X_test)
mse = mean_squared_error(y_test,ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % np.sqrt(mse))
print("Mean Absolute Error : " + str(mean_absolute_error(ypred, y_test)))
print(“R square: ”   )
print(model.score(X, y), 1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1))

#visualize the original and predicted test data in a plot
x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

#plot feature importance
xgb.plot_importance(model)

#plots for different importance types
importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
for  f in importance_types:
	impPlot=model.get_booster().get_score(importance_type=f)
	plt.title("Importance by " + f)
	plt.xlabel("Relative Importance")
	plt.ylabel("Features")
	plt.barh(range(len(impPlot)),list(impPlot.values()))
	plt.yticks(range(len(impPlot)),list(impPlot.keys()))   
	plt.show()
    
#store results in csv
test_pred=np.expm1(model.predict(X_test))
X_test["Course.Grade.with.No.extra.credit"] = np.log1p(test_pred)
X_test.to_csv("xgbresults.csv", columns=["Course.Grade.with.No.extra.credit"], index=False)
