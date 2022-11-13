import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.ensemble import RandomForestClassifier

# Image manipulation libraries
import cv2

DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "smiling"

ENABLE_RESIZE = False
RESIZE_SCALING = 0.5

def loadImgData(dataset_path, img_names):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + file_names[i])

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(image = img_gray, threshold1 = 500, threshold2 = 800) # Canny Edge Detection

        if ENABLE_RESIZE:
            h, w = edges.shape

            w = int(w * RESIZE_SCALING)
            h = int(h * RESIZE_SCALING)

            edges = cv2.resize(edges, (w, h), interpolation = cv2.INTER_LINEAR)

        edges_bin = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow("AAAA", edges_bin)
        cv2.waitKey(0)

        return

        img_proc = np.array(edges_bin, dtype = 'uint8').flatten()
        
        img_data.append(img_proc)

    img_data = np.array(img_data)

    return img_data


# TRAINING

label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")
file_names = label_file[LABEL_IMG_NAMES].values

X_train = loadImgData(DATASET_PATH, file_names)
y_train = label_file[LABEL_NAME].values

clf = RandomForestClassifier(random_state = 42)

clf.fit(X_train, y_train)

# TESTING

label_file = pd.read_csv(TEST_DATASET_PATH + '/labels.csv', delimiter = "\t")
file_names = label_file[LABEL_IMG_NAMES].values

X_test = loadImgData(TEST_DATASET_PATH, file_names)
y_test = label_file[LABEL_NAME].values

predicted = clf.predict(X_test)

print("Labels: ", y_test)
print("Predicted: ", predicted)

print("Score: ", clf.score(X_test, y_test))
