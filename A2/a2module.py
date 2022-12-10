#%% Import libraries
# System libraries
import time

# Data manipulation libraries
import numpy as np
import pandas as pd

# Image handling libraries
import cv2

# Sklearn libraries
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "smiling"

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
def load_data_source(dataset_path, img_names, use_grayscale = True):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        h, w = 0, 0

        if use_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = img.shape

        else:
            h, w, _ = img.shape

        # Crop the image
        img = img[int(1 * h / 5):int(4 * h / 5), int(2 * w / 8):int(6 * w / 8)]

        # Normalize the image so that values stay low
        img = np.array(img, dtype = np.single).flatten() / 255.0
        
        img_data.append(img)

    img_data = np.array(img_data)

    return img_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path, use_grayscale):
    # Read the csv file and extract label_file for each image
    label_file = pd.read_csv(dataset_path + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    X = load_data_source(dataset_path, file_names, use_grayscale)

    y = label_file[LABEL_NAME].values
    y = LabelEncoder().fit_transform(y)

    return X, y


# A function to run the code to solve the task A1
def run_task(use_grayscale = True):
    #%% Select the classifiers
    print("Setting up classifiers...", end = " ")

    parameters = {
        'learning_rate': ['optimal'],
        'random_state': [42],
        'alpha': [1e-5, 1e-4],
        'loss': ['hinge', 'log_loss', 'perceptron'],
        'penalty': ['l1', 'l2'],
        'max_iter': [3000]
    }

    clf_grid = GridSearchCV(SGDClassifier(), parameters, scoring = ('f1'), cv = 5, refit = True, n_jobs = -1, verbose = 2)

    print("Done\n")


    #%% Load training data
    timer = Timer()
    print("Loading in training data...", end = " ")
    X_train, y_train = load_Xy_data(DATASET_PATH, use_grayscale)
    print("Done in " + timer.print() + "s\n")


    #%% Cross-validation and fitting the best model
    timer.reset()
    print("Performing cross-validation of all the models and training the best model on all the data...", end = " ")
    clf_grid.fit(X_train, y_train)
    print("Done in " + timer.print() + "s\n")


    #%% Load test data
    timer.reset()
    print("Loading in test data...", end = " ")
    X_test, y_test = load_Xy_data(TEST_DATASET_PATH, use_grayscale)
    print("Done in " + timer.print() + "s\n")


    #%% Testing
    print("Obtaining model y_pred\n")
    y_pred = clf_grid.predict(X_test) 


    #%% Print the results
    print("Results:\n")
    print("Labels: ", y_test)
    print("Predicted: ", y_pred)
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
    
    # Print the classification report 
    print(classification_report(y_test, y_pred)) 

    # Print cross validation scores for the whole grid
    print("Mean cross-validation test scores: ", clf_grid.cv_results_["mean_test_score"])

    # Print the best params in the grid
    print("Best score: %0.3f" % (clf_grid.best_score_))
    print("Best parameters set:")
    best_parameters = clf_grid.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# Execute the code if the script is run on its own
if __name__ == "__main__":
    run_task()