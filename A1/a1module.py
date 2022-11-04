import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn import metrics

# Image manipulation libraries
from PIL import Image

# Plotting libraries
import matplotlib.pyplot as plt


# Read the csv file
img_labels = pd.read_csv('./Datasets/cartoon_set/labels.csv', delimiter = "\t")
print(img_labels.shape)
