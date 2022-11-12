import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Image manipulation libraries
import cv2

# Plotting libraries
import matplotlib.pyplot as plt


DATASET_PATH = './Datasets/cartoon_set'
LABEL_IMG_NAMES = "file_name"


label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")
labels = label_file[LABEL_IMG_NAMES].values

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 300
params.maxArea = 3000

params.filterByCircularity = True
params.minCircularity = 0.2
 
params.filterByInertia = True
params.minInertiaRatio = 0.3

params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)


col_data = np.zeros((len(labels), 6))

for i in range(len(labels)):
    img = cv2.imread(DATASET_PATH + '/img/' + labels[i])

    keypoints = detector.detect(img)

    # Find average color and its mean std dev in the blobs
    mask_circle = np.zeros(img.shape[:2], np.uint8)

    for j in range(len(keypoints)):
        pt = keypoints[j].pt
        radius = keypoints[j].size / 2
        mask_circle = cv2.circle(mask_circle, center = (int(pt[0]), int(pt[1])), radius = int(radius), color = (255, 255, 255), thickness = -1)

    mean, std = cv2.meanStdDev(img, mask = mask_circle)

    col_data[i, 0:3] = mean[:, 0]
    col_data[i, 3:6] = std[:, 0]

df = pd.DataFrame(data = col_data, columns = ['B', 'G', 'R', 'S_B', 'S_G', 'S_R'])

df.to_csv('./B2/col_data.csv', sep = "\t")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img_proc = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 
# cv2.imshow("Keypoints", img_proc)
# cv2.waitKey(0)