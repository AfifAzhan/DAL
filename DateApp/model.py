# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:17:45 2022

@author: RAN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 21:12:30 2022
@author: Nadeeya Norjemee
This svm model is using only important features
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#read data
df = pd.read_excel('svm_imp_feat_unbal.xlsx')

x = df.iloc[:, :9] # x - the 9 features
y = df.iloc[:, 9] # y - the date class

"""Splitting data"""
#The dataset is split into 70:30 , train:test ratio with a random state of 0
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3, random_state=0)

"""Standardize the data"""
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

"""import model svm"""
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=10, gamma=1)
#fit the model with the data
svm.fit(x_train, y_train)

#When the code is running for the first time, the commented GridSearch code below was run to 
#find the best parameters for the model

"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
param_grid = {'C' : [0.1, 1, 10 , 100, 1000],
              'gamma':[1,0.1, 0.01 , 0.001, 0.0001],
              'kernel':['rbf', 'linear']}
grid = GridSearchCV(SVC() ,param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)
#fitting the best model for grid search
print(grid.best_params_)
#print how our model looks after hyper- parameter tuning
print(grid.best_estimator_)
"""
#Once we get the best parameters, we use it to fit in our model and 
#comment out the above GridSearch code

"""Evaluating model performance"""

from sklearn.metrics import accuracy_score
y_pred = svm.predict(x_train)
print("Training Accuracy :" , accuracy_score(y_train, y_pred)*100)
y_pred_test = svm.predict(x_test)
print("Testing Accuracy :" , accuracy_score(y_test, y_pred_test)*100)

#Visualize the confusion matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score 
cm  = confusion_matrix(y_test, y_pred_test) 
cm_df = pd.DataFrame(cm,index = np.unique(y), columns =np.unique(y)) 
plt.subplots(figsize=(20,15)) 
sns.heatmap(cm_df, annot=True, 
cmap="YlGnBu",fmt=",",annot_kws={"size": 13}) 
plt.title('SVM important features Algorithm \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred_test))) 
plt.ylabel('y_test') 
plt.xlabel('y_test_pred') 
plt.show() 

#Classification report
from sklearn.metrics import classification_report
grid_predictions = svm.predict(x_test)
#print classification report
print(classification_report(y_test, grid_predictions))

# Import model acquisition library
import pickle

pickle.dump(svm, open('model1.pkl','wb'))

model = pickle.load(open('model1.pkl','rb'))
print("The date type is: ",model.predict([x_test[0]]))

print(x_test[0])

# ^ Model created










