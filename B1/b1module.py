import numpy as np
import pandas as pd

# Image manipulation libraries
import cv2

DATASET_PATH = './Datasets/cartoon_set'
LABEL_IMG_NAMES = "file_name"
LABEL_NAME = "face_shape"


label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")
file_names = label_file[LABEL_IMG_NAMES].values
labels = label_file[LABEL_NAME].values

for i in range(1):
    img = cv2.imread(DATASET_PATH + '/img/' + file_names[5])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    edges = cv2.Canny(image = img_gray, threshold1 = 500, threshold2 = 800) # Canny Edge Detection

    edges_bin = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]
    #edges = cv2.GaussianBlur(edges, (3, 3), 0) 

    #scale = 0.5

    #width = int(edges.shape[1] * scale)
    #height = int(edges.shape[0] * scale)

    #edges = cv2.resize(edges, (width, height))
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges_bin)
    cv2.waitKey(0)
