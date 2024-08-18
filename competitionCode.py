import numpy as np
import pandas as pd
import os

import tensorflow as tf
from sklearn import preprocessing


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df = train_df.dropna()

Y_train = train_df["Survived"]

train_df = train_df[["PassengerId","Sex", "Age", "SibSp", "Parch", "Fare"]]
train_df['Sex'].replace(['female','male'], [0,1],inplace=True)
X_train = train_df
print(train_df.head(5))

test_df = test_df[["PassengerId", "Sex", "Age", "SibSp", "Parch", "Fare"]]
test_df['Sex'].replace(['female','male'], [0,1],inplace=True)
X_test = test_df
print(test_df.head(5))

from sklearn import preprocessing


X_test = X_test.fillna(0)
'''
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)
'''

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")