import numpy as np
import pandas as pd
import os

import tensorflow as tf
from sklearn import preprocessing


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df = train_df.dropna()

Y_train = train_df["Survived"]

train_df = train_df[["Sex", "Age", "Pclass", "SibSp","Parch", "Fare", "Embarked"]]
train_df['Sex'].replace(['female','male'], [0,1],inplace=True)
train_df['Embarked'].replace(['C','S', 'Q'], [0,1,2],inplace=True)

X_train = train_df

X_test = test_df[["Sex", "Age", "Pclass", "SibSp","Parch", "Fare", "Embarked"]]
X_test['Sex'].replace(['female','male'], [0,1],inplace=True)
X_test['Embarked'].replace(['C','S', 'Q'], [0,1,2],inplace=True)
X_test = X_test.fillna(0)

'''
We can use X_train and Y_train to train any model
And used X_test to make predictions
'''



'''
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.Pclass *=10
X_test.Pclass *=10
X_train.Sex *=10
X_test.Sex *=10
X_train = sc.fit_transform(X_train)
X_test_trans = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, Y_train, epochs = 500, batch_size=128)
'''
param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'batch_size': [16, 32, 64],
    'epochs': [30, 60, 90]
}

from sklearn.model_selection import GridSearchCV
# Use GridSearchCV to find the best combination of hyperparameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="accuracy")
grid_result = grid.fit(X_train, Y_train)

# Display the best hyperparameters and corresponding accuracy
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Accuracy: {grid_result.best_score_}")

predictions = grid.predict(X_test)
'''

predictions = classifier.predict(X_test)
##print(predictions)
y = []
for val in range(0, len(predictions)):
  if predictions[val] > 0.5:
    y.append(1)
  else:
    y.append(0)
print(y)
print(X_test.head(5))
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")