import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Image manipulation libraries
import cv2

# Plotting libraries
import matplotlib.pyplot as plt


# Constants
DATASET_PATH = './B2/'
LABEL_NAME = "eye_color"
TEST_SIZE = 0.1


### LEARNING

# Read the csv file and extract label_file for each image
col_data = pd.read_csv(DATASET_PATH + '/col_data.csv', delimiter = "\t")

labels = np.array(col_data[LABEL_NAME].values, dtype = 'uint8')

X = col_data.to_numpy()
X = X[:, 0:6] # Remove the label column


# Define the classifier and the param grid
clf = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 5)}

clf_grid = GridSearchCV(clf, param_grid, cv = 5)


X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state = 1230, test_size = TEST_SIZE)

# Learn the digits on the train subset
clf_grid.fit(X_train, y_train)


### TESTING

print(clf_grid.best_params_) 
grid_predictions = clf_grid.predict(X_test) 
   
# print classification report 
print(classification_report(y_test, grid_predictions)) 