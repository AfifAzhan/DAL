# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:17:45 2022

@author: RAN
"""

# Import dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import classification_report

df = pd.read_excel('svm_imp_feat_unbal.xlsx')
x = df.iloc[:, :9]
y = df.iloc[:, 9]


sc = StandardScaler()

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3, random_state=0)

# Balancing
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)

X_res, y_res = over_sampler.fit_resample(x_train, y_train)

from collections import Counter

print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")

print("x_test", x_train.count())
print("X_res ",X_res.count())


x_train_sd = sc.fit_transform(X_res)
x_test_sd = sc.fit_transform(x_test)


# Balancing done
svm = SVC(kernel='linear', C=10, gamma=1)
svm.fit(x_train_sd, y_res) #changed y_train to y_res

y_pred = svm.predict(x_train)
print("Training Accuracy :" , accuracy_score(y_train, y_pred)*100)


y_pred_test = svm.predict(x_test)
print("Testing Accuracy :" , accuracy_score(y_test, y_pred_test)*100)

cm  = confusion_matrix(y_test, y_pred_test) 
cm_df = pd.DataFrame(cm,index = np.unique(y), columns =np.unique(y)) 
plt.subplots(figsize=(20,15)) 
sns.heatmap(cm_df, annot=True, 
cmap="YlGnBu",fmt=",",annot_kws={"size": 13}) 
plt.title('SVM important features Algorithm \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred_test))) 
plt.ylabel('y_test') 
plt.xlabel('y_test_pred') 
plt.show() 


grid_predictions = svm.predict(x_test_sd)

#print classification report
print(classification_report(y_test, grid_predictions))

user_input =  [296.504,0.7847,1.4898,0.8574,24.9323,-0.5385,3.9749,3.0696,68.7153]

"""Standardize user's input"""
#XNormed = (X - X.mean())/(X.std())
standardized_input =[]
 
i = 0
cols = x.columns
for col in cols:
    inputStd = (user_input[i] - x_train[col].mean())/(x_train[col].std())
    i+=1
    standardized_input.append(inputStd)

# Import model acquisition library
import pickle

pickle.dump(svm, open('model3.pkl','wb'))

model = pickle.load(open('model3.pkl','rb'))
print(model.predict([standardized_input])) 


# ^ Model created

print("Mean: ",x_train[col].mean(), "STD: ", x_train[col].std())
print("standardized:", standardized_input)
print("x_train: ",x_train_sd)
print("X.col: ",cols)







