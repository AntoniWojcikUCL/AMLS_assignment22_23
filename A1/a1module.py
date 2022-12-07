#%% Import libraries
# Data manipulation libraries
import numpy as np
import pandas as pd

# Image handling libraries
import cv2

# Sklearn libraries
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "gender"

#%% Helper functions

# Load images, preprocess and flatten them, and combine into an array
def load_data_source(dataset_path, img_names):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        # Normalize the image so that values stay low
        img = np.array(img, dtype = np.single).flatten() / 255.0
        
        img_data.append(img)

    img_data = np.array(img_data)

    return img_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path):
    # Read the csv file and extract label_file for each image
    label_file = pd.read_csv(dataset_path + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    X = load_data_source(dataset_path, file_names)

    y = label_file[LABEL_NAME].values
    y = LabelEncoder().fit_transform(y)

    return X, y


# A function to run the code to solve the task A1
def run_task():
    #%% Select the classifiers
    print("Setting up classifiers...", end = " ")
    clf = []
    clf.append(
        SGDClassifier(learning_rate = 'optimal', alpha = 1e-5, penalty = 'l1', max_iter = 3000, shuffle = True, loss = 'perceptron', random_state = 42, n_jobs = -1, verbose = 0)
    )
    clf.append(
        SVC(gamma = "auto", random_state = 42, verbose = 0)
    )
    print("Done\n")

    #%% Load training data
    print("Loading in training data...", end = " ")
    X_train, y_train = load_Xy_data(DATASET_PATH)
    print("Done\n")


    #%% Cross-validation
    print("Selecting cross validation data...", end = " ")
    X_val, _, y_val, _ = train_test_split(X_train, y_train, test_size = 0.8, random_state = 42)
    print("Done\n")

    cv_score_min = float("inf")
    cv_score_idx_min = 0

    for i in range(len(clf)):
        print("Performing cross-validation of model " + str(i) + "...", end = " ")
        cv_scores = cross_val_score(clf[i], X_val, y_val, scoring = ('f1'), cv = KFold(n_splits = 5), n_jobs = 5, verbose = 2)
        print("Done\n")

        mean_score = np.mean(cv_scores)

        if cv_score_min > mean_score:
            cv_score_idx_min = i
            cv_score_min = mean_score

        print("K-fold cross validation scores: ", cv_scores)
        print("Mean score: ", mean_score, "\n")

    print("Cross-validation done. Best model: " + str(cv_score_idx_min) + "\n")

    clf_optimal = clf[cv_score_idx_min]


    #%% Train the best model
    print("Training the best model...")
    clf_optimal.fit(X_train, y_train)
    print("Done\n")


    #%% Load test data
    print("Loading in test data...", end = " ")
    X_test, y_test = load_Xy_data(TEST_DATASET_PATH)
    print("Done\n")


    #%% Testing
    print("Obtaining model predictions\n")
    predictions = clf_optimal.predict(X_test) 


    #%% Print the results
    print("Results:\n")
    print("Labels: ", y_test)
    print("Predicted: ", predictions)
    print("Score:", clf_optimal.score(X_test, y_test))
    
    # Print the classification report 
    print(classification_report(y_test, predictions)) 


# Execute the code if the script is run on its own
if __name__ == "__main__":
    run_task()