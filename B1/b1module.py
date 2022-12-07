#%% Import libraries
# System libraries
import time

# Data manipulation libraries
import numpy as np
import pandas as pd

# Image handling libraries
import cv2

# Sklearn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/cartoon_set'
TEST_DATASET_PATH = './Datasets/cartoon_set_test'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "face_shape"

ENABLE_EDGE_DETECTION = False
ENABLE_RESIZE = True
RESIZE_SCALING = 0.5

#%% Helper functions and classes
class Timer:
    timer = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.timer = time.time()

    def print(self):
        return str(time.time() - self.timer)

# Load images, preprocess and flatten them, and combine into an array
def load_data_source(dataset_path, img_names):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if ENABLE_EDGE_DETECTION:
            edges = cv2.Canny(image = img_gray, threshold1 = 500, threshold2 = 800) # Canny Edge Detection

            if ENABLE_RESIZE:
                h, w = edges.shape

                w = int(w * RESIZE_SCALING)
                h = int(h * RESIZE_SCALING)

                edges = cv2.resize(edges, (w, h), interpolation = cv2.INTER_LINEAR)

            edges_bin = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]

            img_proc = np.array(edges_bin, dtype = 'uint8').flatten()
        
            img_data.append(img_proc)

        else:
            img_gray = np.array(img_gray, dtype = np.single).flatten() / 255.0
        
            img_data.append(img_gray)

    img_data = np.array(img_data)

    return img_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path):
    # Read the csv file and extract label_file for each image
    label_file = pd.read_csv(dataset_path + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    X = load_data_source(dataset_path, file_names)

    y = label_file[LABEL_NAME].values

    return X, y


# A function to run the code to solve the task A1
def run_task():
    #%% Select the classifiers
    print("Setting up classifiers...", end = " ")
    clf = RandomForestClassifier(random_state = 42, criterion = "entropy", min_samples_split = 20, n_estimators = 100, n_jobs = -1, verbose = True)
    print("Done\n")


    #%% Load training data
    print("Loading in training data...", end = " ")
    X_train, y_train = load_Xy_data(DATASET_PATH)
    print("Done\n")


    #%% Train the model
    print("Training the model...")
    timer = Timer()
    clf.fit(X_train, y_train)
    print("Done in " + timer.print() + "s\n")


    #%% Load test data
    print("Loading in test data...", end = " ")
    X_test, y_test = load_Xy_data(TEST_DATASET_PATH)
    print("Done\n")


    #%% Testing
    print("Obtaining model predictions\n")
    predictions = clf.predict(X_test) 


    #%% Print the results
    print("Results:\n")
    print("Labels: ", y_test)
    print("Predicted: ", predictions)
    print("Score:", clf.score(X_test, y_test))
    
    # Print the classification report 
    print(classification_report(y_test, predictions)) 


# Execute the code if the script is run on its own
if __name__ == "__main__":
    run_task()