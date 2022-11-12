import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Image manipulation libraries
import cv2

# Plotting libraries
import matplotlib.pyplot as plt


# Constants
DATASET_PATH = './B2/'
LABEL_NAME = "eye_color"
TEST_SIZE = 0.3


### LEARNING

# Read the csv file and extract label_file for each image
col_data = pd.read_csv(DATASET_PATH + '/col_data.csv', delimiter = "\t")

labels = np.array(col_data[LABEL_NAME].values, dtype = 'uint8')

X = col_data.to_numpy()#dtype = "uint8")
X = X[:, 0:6]

# Select the classifier 
clf = RandomForestClassifier(random_state=0)


X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state = 1230, test_size = TEST_SIZE)

# Learn the digits on the train subset
clf.fit(X_train, y_train)


### TESTING

y_predicted = clf.predict(X_test)

# Print the results
print("Labels: ", y_test)
print("Predicted: ", y_predicted)
print("Score: ", clf.score(X_test, y_test))