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

incorr_imgs = 0

#for i in range(0, 9999):
#
#    img = cv2.imread(DATASET_PATH + '/img/' + labels[0])
#
#    keypoints = detector.detect(img)
#
#    if len(keypoints) != 2:
#        incorr_imgs += 1
#        break
#
#print(i)

img = cv2.imread(DATASET_PATH + '/img/' + labels[3])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = detector.detect(img)

col_mean = np.zeros((4))

mask_empty = np.zeros(img.shape[:2], np.uint8)

for i in range(len(keypoints)):
    pt = keypoints[i].pt
    mask_circle = cv2.circle(mask_empty, center = (int(pt[0]), int(pt[1])), radius = int(keypoints[i].size), color = (255, 255, 255), thickness = -1)

    col_mean += np.asarray(cv2.mean(img, mask = mask_circle))

col_mean = col_mean[0:3]

col_mean /= len(keypoints)

print(col_mean)

img_proc = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", img_proc)
cv2.waitKey(0)