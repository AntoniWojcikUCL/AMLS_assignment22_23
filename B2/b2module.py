#%% Import libraries
# System libraries
import time

# Data manipulation libraries
import numpy as np
import pandas as pd

# Image handling libraries
import cv2
import matplotlib.pyplot as plt

# Sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/cartoon_set'
TEST_DATASET_PATH = './Datasets/cartoon_set_test'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "eye_color"

DEF_DEBUG_IMG_PREVIEW = -1 # Type -1 if none of the images should be shown

#%% Helper functions and classes
class Timer:
    timer = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.timer = time.time()

    def print(self):
        return str(time.time() - self.timer)

# Load image data and preprocess it to extract mean eye colors and their std dev for each image 
# and store the output in a 6 x [image number] array. 
def load_data_source(dataset_path, file_names):

    # Preapre an array used to store eye color information gathered from the images
    # We are extracting average pixel color in the eye vicinity and its standard deviation
    col_data = np.zeros((len(file_names), 7))

    # Define the mask array
    mask = []

    # Find average color and its mean std dev in the blobs for each image
    for i in range(len(file_names)):
        img = cv2.imread(dataset_path + '/img/' + file_names[i])

        # Generate the hard-coded mask based on the first image
        if i == 0:
            mask = np.zeros(img.shape[:2], np.uint8)

            radius = 18
            pt = (294, 260)
            mask = cv2.circle(mask, center = pt, radius = radius, color = (255, 255, 255), thickness = -1)
            pt = (205, 260)
            mask = cv2.circle(mask, center = pt, radius = radius, color = (255, 255, 255), thickness = -1)

        # Find mean and std dev of pixel color in the mask
        mean, std = cv2.meanStdDev(img, mask = mask)

        if i == DEF_DEBUG_IMG_PREVIEW:
            cv2.imshow("Preview of the mask", cv2.bitwise_and(img, img, mask = mask))
            cv2.waitKey(0)

        # Store the data in the array
        col_data[i, 0:3] = mean[:, 0]
        col_data[i, 3:6] = std[:, 0]
        col_data[i, 6] = np.any(mean[:, 0] > 55) # If true - probably not sunglasses

    return col_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path, remove_sunglasses_datapoints = False, add_sunglasses_labels = False):
    label_file = pd.read_csv(dataset_path  + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    X = load_data_source(dataset_path, file_names)
    y = label_file[LABEL_NAME].values

    if add_sunglasses_labels:
        eyes_not_visible = (X[:, 6] == 0)
        y[eyes_not_visible] = 5

    if (not add_sunglasses_labels) and remove_sunglasses_datapoints:
        eyes_visible = (X[:, 6] > 0)

        y = y[eyes_visible]
        X = X[eyes_visible, 0:6]

    else:
        X = X[:, 0:6]

    return X, y

# Function to generate a convergence plot for the model
def plot_convergence(clf, X, y, plot_out_path = ""):
    font = {'size' : 12}
    plt.rc('font', **font)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(bottom=0.2, left=0.2)

    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": KFold(n_splits=20),
        "score_type": "both",
        "n_jobs": -1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    LearningCurveDisplay.from_estimator(clf, **common_params, ax=ax)
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve for {clf.__class__.__name__}")

    if plot_out_path:
        plt.savefig(plot_out_path)


# A function to run the code to solve the task A1
def run_task(add_sunglasses_lab = False, rm_train_sun_dp = True, rm_test_sun_dp = True, gen_convergence_plot = False, plot_out_path = ""):
    # If we want to add new labels for sunglasses to the data, then we shouldn't remove these datapoints from training and testing
    if add_sunglasses_lab:
        rm_train_sun_dp = False
        rm_test_sun_dp = False

    #%% Define the classifier and the param grid
    print("Setting up classifiers...", end = " ")

    param_grid = {'n_neighbors': np.arange(1, 10)}

    clf_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, n_jobs = -1, verbose = 2)
    print("Done\n")


    #%% Load training data
    timer = Timer()
    print("Loading in training data...", end = " ")
    X_train, y_train = load_Xy_data(DATASET_PATH, rm_train_sun_dp, add_sunglasses_lab)
    print("Done in: " + timer.print() + "s\n")


    #%% Cross-validation to tune hyperparams and fit the best model
    timer.reset()
    print("Training the best model...")
    clf_grid.fit(X_train, y_train)
    print("Done in: " + timer.print() + "s\n")


    #%% Use cross-validation to generage a convergence plot for the model with best hyperparameters
    if gen_convergence_plot:
        timer.reset()
        print("Generating a convergence plot...", end = " ")
        plot_convergence(clf_grid.best_estimator_, X_train, y_train, plot_out_path)
        print("Done in: " + timer.print())


    #%% Load test data
    timer.reset()
    print("Loading in test data...", end = " ")
    X_test, y_test = load_Xy_data(TEST_DATASET_PATH, rm_test_sun_dp, add_sunglasses_lab)
    print("Done in: " + timer.print() + "s\n")

    #%% Testing
    print("Obtaining model predictions\n")
    y_pred = clf_grid.predict(X_test)
    
    #%% Print the results

    # Print the best value for K in KNN
    print("Optimal number of nearest neighbours for KNN: " + str(clf_grid.best_params_["n_neighbors"]) )
    # Print confusion matrix
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
    # Print classification report 
    print(classification_report(y_test, y_pred)) 


# Execute the code if the script is run on its own
if __name__ == "__main__":
    run_task()