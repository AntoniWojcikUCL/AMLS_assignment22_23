import sys 
import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Image manipulation libraries
import cv2


DATASET_PATH = './Datasets/cartoon_set'
TEST_DATASET_PATH = './Datasets/cartoon_set_test'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "eye_color"

READ_DATA = True
SAVE_DATA = False
DEBUG_EYE_DETECTION = False
REMOVE_TEST_INVISIBLE_DATAPOINTS = True


# Load image data and preprocess it to extract mean eye colors and their std dev for each image 
# and store the output in a 6 x [image number] array. Also enable saving data to a .csv file
def loadImgData(dataset_path, file_names, out_file_name):

    # Preapre an array used to store eye color information gathered from the images
    # We are extracting average pixel color in the eye vicinity and its standard deviation
    col_data = np.zeros((len(file_names), 7))

    if READ_DATA:
        col_data = pd.read_csv('./B2/preprocessed_data/' + out_file_name, delimiter = "\t").to_numpy()
    
    else:
        # Tune blob detector params to enable eye detection in the images
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 300
        params.maxArea = 3000

        params.filterByCircularity = True
        params.minCircularity = 0.2
        
        params.filterByInertia = True
        params.minInertiaRatio = 0.24

        params.filterByConvexity = False

        # Create an instance of the blob detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Find average color and its mean std dev in the blobs for each image
        for i in range(len(file_names)):
            img = cv2.imread(dataset_path + '/img/' + file_names[i])

            # Find locations of the eyes and their approximate sizes
            keypoints = detector.detect(img)

            if DEBUG_EYE_DETECTION:
                img_debug = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
                # Show keypoints
                cv2.imshow("Keypoints", img_debug)
                cv2.waitKey(0)

                exit()

            # Create circular masks around the eyes and combine them
            mask_circle = np.zeros(img.shape[:2], np.uint8)

            for j in range(len(keypoints)):
                pt = keypoints[j].pt
                radius = keypoints[j].size / 2
                mask_circle = cv2.circle(mask_circle, center = (int(pt[0]), int(pt[1])), radius = int(radius), color = (255, 255, 255), thickness = -1)

            # Find mean and std dev of pixel color in the mask
            mean, std = cv2.meanStdDev(img, mask = mask_circle)

            # Store the data in the array
            col_data[i, 0:3] = mean[:, 0]
            col_data[i, 3:6] = std[:, 0]
            col_data[i, 6] = np.any(mean[:, 0] > 55) # If true - probably not sunglasses

        if SAVE_DATA:
            # Store the color array in a pandas data frame and save it to a file
            df = pd.DataFrame(data = col_data, columns = ['B', 'G', 'R', 'S_B', 'S_G', 'S_R', 'Visible'])

            df.to_csv('./B2/preprocessed_data/' + out_file_name, sep = "\t", index = False)

    return col_data

label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")
file_names = label_file[LABEL_IMG_NAMES].values

X_train = loadImgData(DATASET_PATH, file_names, "col_data.csv")
eyes_visible = (X_train[:, 6] > 0)
X_train = X_train[eyes_visible, 0:6]
y_train = label_file[LABEL_NAME].values
y_train = y_train[eyes_visible]

# Define the classifier and the param grid
clf = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 10)}

clf_grid = GridSearchCV(clf, param_grid, cv = 5, n_jobs = -1, verbose = 2)


# Train the model
clf_grid.fit(X_train, y_train)

predict_fit_backwards = clf_grid.predict(X_train) 

idx_wrong = (y_train != predict_fit_backwards)

np.set_printoptions(threshold = sys.maxsize)
print("Failures in predictions in training data: \n", np.where(idx_wrong))

idx_correct = (idx_wrong == False)

# Retrain the model on datapoints without sunglasses (presumably; I know this is not entirely true)
clf_grid.fit(X_train[idx_correct], y_train[idx_correct])


### TESTING

label_file = pd.read_csv(TEST_DATASET_PATH + '/labels.csv', delimiter = "\t")
file_names = label_file[LABEL_IMG_NAMES].values

X_test = loadImgData(TEST_DATASET_PATH, file_names, "col_data_test.csv")
y_test = label_file[LABEL_NAME].values

if REMOVE_TEST_INVISIBLE_DATAPOINTS:
    eyes_visible = (X_test[:, 6] > 0)
    X_test = X_test[eyes_visible, 0:6]
    y_test = y_test[eyes_visible]
else:
    X_test = X_test[:, 0:6]

print(clf_grid.best_params_) 
grid_predictions = clf_grid.predict(X_test) 
   
# Print classification report 
print(classification_report(y_test, grid_predictions)) 