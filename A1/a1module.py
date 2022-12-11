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
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "gender"

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
def load_data_source(dataset_path, img_names, use_grayscale = True, show_mean = False, y_labels = np.zeros(0)):
    img_data = []

    # Loop over images in the data set
    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        h, w = 0, 0

        # Transform to grayscale if flag true
        if use_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = img.shape

        else:
            h, w, _ = img.shape

        # Crop the image
        img = img[int(1 * h / 10):int(9 * h / 10), int(1 * w / 8):int(7 * w / 8)]

        # Append images
        img_data.append(np.array(img, dtype = np.single))

    img_data = np.array(img_data)

    # Display mean and std of images with the same label
    if show_mean:
        for i in range(len(np.unique(y_labels))):
            img_idx = (y_labels == i)
            img_mean = np.mean(img_data[img_idx, :, :], axis = 0)
            cv2.imshow("Mean of images with label " + str(i), np.array(img_mean, dtype = np.uint8))
            img_std = np.std(img_data[img_idx, :, :], axis = 0)
            cv2.imshow("STD of images with label " + str(i), np.array(img_std, dtype = np.uint8))
        
        cv2.waitKey(1)

    # Reshape a stack of 2D images (3D array) to a 2D array of flatten image data per row
    img_data = img_data.reshape(img_data.shape[0], img_data.shape[1] * img_data.shape[2])

    # Normalize the image so that values stay low
    img_data /= 255.0

    return img_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path, use_grayscale, show_mean = False):
    # Read the csv file and extract label_file for each image
    label_file = pd.read_csv(dataset_path + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    y = label_file[LABEL_NAME].values
    y = LabelEncoder().fit_transform(y)

    X = load_data_source(dataset_path, file_names, use_grayscale, show_mean, y)

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
def run_task(run_cross_val = True, clf_optimal_idx = 0, use_grayscale = True, show_mean = False, gen_convergence_plot = False, plot_out_path = ""):
    #%% Select the classifiers
    print("Setting up classifiers...", end = " ")
    clf = []
    clf.append(
        SGDClassifier(loss = 'perceptron', penalty = 'l1', learning_rate = 'optimal', alpha = 1e-4, max_iter = 3000, shuffle = True, random_state = 42, n_jobs = -1, verbose = 0)
    )
    clf.append(
        LogisticRegression(solver = 'saga', penalty = 'l1', max_iter = 3000, random_state = 42, n_jobs = -1, verbose = 2)
    )
    print("Done\n")


    #%% Load training data
    timer = Timer()
    print("Loading in training data...", end = " ")
    X_train, y_train = load_Xy_data(DATASET_PATH, use_grayscale, show_mean)
    print("Done in " + timer.print() + "s\n")


    #%% Cross-validation
    if run_cross_val:
        print("Selecting cross validation data...", end = " ")
        X_val, _, y_val, _ = train_test_split(X_train, y_train, test_size = 0.6, random_state = 42)
        print("Done\n")

        cv_score_best = float("-inf")

        timer.reset()
        for i in range(len(clf)):
            print("Performing cross-validation of model " + str(i) + "...", end = " ")
            cv_scores = cross_val_score(clf[i], X_val, y_val, scoring = ('f1'), cv = KFold(n_splits = 5), n_jobs = 5, verbose = 2)
            print("Done\n")

            mean_score = np.mean(cv_scores)

            if cv_score_best < mean_score:
                clf_optimal_idx = i
                cv_score_best = mean_score

            print("K-fold cross validation scores: ", cv_scores)
            print("Mean score: ", mean_score, "\n")

        print("Cross-validation done in " + timer.print() + "s.\nBest model: " + str(clf_optimal_idx) + "\n")

    clf_optimal = clf[clf_optimal_idx]


    #%% Train the best model
    timer.reset()
    print("Training the best model...")
    clf_optimal.fit(X_train, y_train)
    print("Done in " + timer.print() + "s\n")


    #%% Use cross-validation to generage a convergence plot for the best model
    if gen_convergence_plot:
        timer.reset()
        print("Generating a convergence plot...", end = " ")
        plot_convergence(clf_optimal, X_train, y_train, plot_out_path)
        print("Done in: " + timer.print())


    #%% Load test data
    timer.reset()
    print("Loading in test data...", end = " ")
    X_test, y_test = load_Xy_data(TEST_DATASET_PATH, use_grayscale)
    print("Done in " + timer.print() + "s\n")


    #%% Testing
    print("Obtaining model predictions\n")
    y_pred = clf_optimal.predict(X_test) 


    #%% Print the results
    print("Results:\n")
    print("Labels: ", y_test)
    print("Predicted: ", y_pred)
    print("Score: ", clf_optimal.score(X_test, y_test))
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
    
    # Print the classification report 
    print(classification_report(y_test, y_pred)) 


# Execute the code if the script is run on its own
if __name__ == "__main__":
    run_task(False, 0, True, True)