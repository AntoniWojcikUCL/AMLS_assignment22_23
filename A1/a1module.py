import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Image manipulation libraries
import cv2



# Constants
DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "gender"

# Helper functions

def loadImgData(dataset_path, img_names):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        # Normalize the image so that values stay low
        img = np.array(img, dtype = np.single).flatten() / 255.0
        
        img_data.append(img)

    img_data = np.array(img_data)

    return img_data


### TRAINING

# Read the csv file and extract label_file for each image
label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")

file_names = label_file[LABEL_IMG_NAMES].values
y_train = label_file[LABEL_NAME].values

X_train = loadImgData(DATASET_PATH, file_names)

y_train = LabelEncoder().fit_transform(y_train)

# Select the classifier 
clf = SGDClassifier(learning_rate = 'optimal', alpha = 1e-5, penalty = 'l1', max_iter = 3000, shuffle = True, loss = 'perceptron', verbose = True, random_state = 42, n_jobs = -1)

# Learn the digits on the train subset
clf.fit(X_train, y_train)


### TESTING

# Read the csv file and extract label_file for each image
label_file = pd.read_csv(TEST_DATASET_PATH + '/labels.csv', delimiter = "\t")

file_names = label_file[LABEL_IMG_NAMES].values
y_test = label_file[LABEL_NAME].values

X_test = loadImgData(TEST_DATASET_PATH, file_names)

y_test = LabelEncoder().fit_transform(y_test)

# Learn the digits on the train subset
predicted = clf.predict(X_test)


# Print the results
print("Labels: ", y_test)
print("Predicted: ", predicted)
print("Score:", clf.score(X_test, y_test))

predictions = clf.predict(X_test) 
   
# Print classification report 
print(classification_report(y_test, predictions)) 