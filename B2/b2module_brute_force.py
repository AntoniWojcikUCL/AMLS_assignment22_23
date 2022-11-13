import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.ensemble import RandomForestClassifier

# Image manipulation libraries
import cv2

# Plotting libraries
import matplotlib.pyplot as plt


# Constants
DATASET_PATH = './Datasets/cartoon_set'
TEST_DATASET_PATH = './Datasets/cartoon_set_test'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "eye_color"

# Helper functions

def loadImgData(dataset_path, img_names):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        w, h, _ = img.shape

        img_proc = img[int(4 * h / 9) : int(3 * h / 5), int(2 * w / 7) : int(w / 2)]

        img_proc = np.array(img_proc).flatten()
        
        img_data.append(img_proc)

    img_data = np.array(img_data)

    return img_data

def unisonShuffleCopies(a, b, seed):
    assert len(a) == len(b)

    p = np.random.RandomState(seed = seed).permutation(len(a))
    return a[p], b[p]



### LEARNING

# Read the csv file and extract label_file for each image
label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")

img_names = label_file[LABEL_IMG_NAMES].values
X_train = loadImgData(DATASET_PATH, img_names)

labels = label_file[LABEL_NAME].values

# Shuffle all the images
y_train, img_names = unisonShuffleCopies(labels, img_names, seed = 42)

# Select the classifier 
clf = RandomForestClassifier(random_state = 42)

# Make the model learn 

# Learn the digits on the train subset
clf.fit(X_train, y_train)


### TESTING

# Read the csv file and extract label_file for each image
label_file = pd.read_csv(TEST_DATASET_PATH + '/labels.csv', delimiter = "\t")

img_names = label_file[LABEL_IMG_NAMES].values
X_test = loadImgData(TEST_DATASET_PATH, img_names)

y_test = label_file[LABEL_NAME].values

# Learn the digits on the train subset
predicted = clf.predict(X_test)


# Print the results
print("Labels: ", y_test)
print("Predicted: ", predicted)

print("Score: ", clf.score(X_test, y_test))