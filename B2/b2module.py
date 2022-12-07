#%% Import libraries
# Data manipulation libraries
import numpy as np
import pandas as pd

# Image handling libraries
import cv2

# Sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/cartoon_set'
TEST_DATASET_PATH = './Datasets/cartoon_set_test'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "eye_color"

REMOVE_TRAIN_INVISIBLE_DATAPOINTS = True
REMOVE_TEST_INVISIBLE_DATAPOINTS = False

#%% Helper functions

# Load image data and preprocess it to extract mean eye colors and their std dev for each image 
# and store the output in a 6 x [image number] array. Also enable saving data to a .csv file
def load_data_source(dataset_path, file_names, out_file_name):

    # Preapre an array used to store eye color information gathered from the images
    # We are extracting average pixel color in the eye vicinity and its standard deviation
    col_data = np.zeros((len(file_names), 7))

    # Define the mask array
    mask_circle = []

    # Find average color and its mean std dev in the blobs for each image
    for i in range(len(file_names)):
        img = cv2.imread(dataset_path + '/img/' + file_names[i])

        # Generate the hard-coded mask based on the first image
        if i == 0:
            mask_circle = np.zeros(img.shape[:2], np.uint8)

            radius = 18
            pt = (294, 260)
            mask_circle = cv2.circle(mask_circle, center = pt, radius = radius, color = (255, 255, 255), thickness = -1)
            pt = (205, 260)
            mask_circle = cv2.circle(mask_circle, center = pt, radius = radius, color = (255, 255, 255), thickness = -1)

        # Find mean and std dev of pixel color in the mask
        mean, std = cv2.meanStdDev(img, mask = mask_circle)

        # Store the data in the array
        col_data[i, 0:3] = mean[:, 0]
        col_data[i, 3:6] = std[:, 0]
        col_data[i, 6] = np.any(mean[:, 0] > 55) # If true - probably not sunglasses

    return col_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path, remove_sunglasses: bool):
    label_file = pd.read_csv(dataset_path  + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    X = load_data_source(dataset_path , file_names, "col_data.csv")
    y = label_file[LABEL_NAME].values

    if(remove_sunglasses):
        eyes_visible = (X[:, 6] > 0)

        y = y[eyes_visible]
        X = X[eyes_visible, 0:6]

    else:
        X = X[:, 0:6]

    return X, y


#%% Define the classifier and the param grid
print("Setting up classifiers...", end = " ")

param_grid = {'n_neighbors': np.arange(1, 10)}

clf_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, n_jobs = -1, verbose = 2)
print("Done\n")


#%% Load train data
print("Loading in training data...", end = " ")
X_train, y_train = load_Xy_data(DATASET_PATH, remove_sunglasses = True)
print("Done\n")


#%% Train the best model
print("Training the best model...")
clf_grid.fit(X_train, y_train)
print("Done\n")

# predict_fit_backwards = clf_grid.predict(X_train) 
# 
# idx_wrong = (y_train != predict_fit_backwards)
# 
# np.set_printoptions(threshold = sys.maxsize)
# print("Failures in predictions in training data: \n", np.where(idx_wrong))
# 
# idx_correct = (idx_wrong == False)
# 
# # Retrain the model on datapoints without sunglasses (presumably; I know this is not entirely true)
# clf_grid.fit(X_train[idx_correct], y_train[idx_correct])


#%% Load test data
print("Loading in test data...", end = " ")
X_test, y_test = load_Xy_data(TEST_DATASET_PATH, remove_sunglasses = REMOVE_TEST_INVISIBLE_DATAPOINTS)
print("Done\n")

#%% Testing
print("Obtaining model predictions\n")
grid_predictions = clf_grid.predict(X_test) 
   
#%% Print the results

# Print the best value for K in KNN
print(clf_grid.best_params_) 
# Print classification report 
print(classification_report(y_test, grid_predictions)) 