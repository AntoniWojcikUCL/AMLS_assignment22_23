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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#%% Constants
DATASET_PATH = './Datasets/cartoon_set'
TEST_DATASET_PATH = './Datasets/cartoon_set_test'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "face_shape"

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
def load_data_source(dataset_path, img_names, enable_edge_detection = False, enable_resize = False, resize_scaling = 1.0, show_mean = False, y_labels = np.zeros(0)):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        # Transform to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape

        # Crop the image
        img = img[int(5 * h / 10):int(9 * h / 10), int(1 * w / 4):int(3 * w / 4)]

        if enable_resize:
            h, w = img.shape

            img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)

        # If flag true, enable edge detection, else only convert to grayscale
        if enable_edge_detection:
            edges = cv2.Canny(image = img, threshold1 = 500, threshold2 = 800) # Canny Edge Detection

            edges_bin = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]

            img = np.array(edges_bin, dtype = np.uint8)

        else:
            img = np.array(img, dtype = np.single)

        # Append images
        img_data.append(img)

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

    if not enable_edge_detection:
        img_data /= 255.0

    return img_data

# Return X, y data of images and labels
def load_Xy_data(dataset_path, enable_edge_detection = False, enable_resize = False, resize_scaling = 1.0, show_mean = False):
    # Read the csv file and extract label_file for each image
    label_file = pd.read_csv(dataset_path + '/labels.csv', delimiter = "\t")
    file_names = label_file[LABEL_IMG_NAMES].values

    y = label_file[LABEL_NAME].values

    X = load_data_source(dataset_path, file_names, enable_edge_detection, enable_resize, resize_scaling, show_mean, y)

    return X, y


# A function to run the code to solve the task A1
def run_task(enable_edge_detection = True, enable_resize = True, resize_scaling = 0.5, show_mean = False):
    #%% Select the classifiers
    print("Setting up classifiers...", end = " ")
    clf = RandomForestClassifier(random_state = 42, criterion = "entropy", min_samples_split = 20, n_estimators = 100, n_jobs = -1, verbose = True)
    print("Done\n")


    #%% Load training data
    timer = Timer()
    print("Loading in training data...", end = " ")
    X_train, y_train = load_Xy_data(DATASET_PATH, enable_edge_detection, enable_resize, resize_scaling, show_mean)
    print("Done in " + timer.print() + "s\n")


    #%% Train the model
    timer.reset()
    print("Training the model...")
    clf.fit(X_train, y_train)
    print("Done in " + timer.print() + "s\n")


    #%% Load test data
    timer.reset()
    print("Loading in test data...", end = " ")
    X_test, y_test = load_Xy_data(TEST_DATASET_PATH, enable_edge_detection, enable_resize, resize_scaling)
    print("Done in " + timer.print() + "s\n")


    #%% Testing
    print("Obtaining model y_pred\n")
    y_pred = clf.predict(X_test) 


    #%% Print the results
    print("Results:\n")
    print("Labels: ", y_test)
    print("Predicted: ", y_pred)
    print("Score: ", clf.score(X_test, y_test))
    print("Confusion matrix: ", confusion_matrix(y_test, y_pred))
    
    # Print the classification report 
    print(classification_report(y_test, y_pred)) 


# Execute the code if the script is run on its own
if __name__ == "__main__":
    run_task(show_mean = True)