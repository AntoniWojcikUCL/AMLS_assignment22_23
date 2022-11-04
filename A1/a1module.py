import numpy as np
import pandas as pd

# Sklearn libraries 
from sklearn import metrics

# Image manipulation libraries
from PIL import Image

# Plotting libraries
import matplotlib.pyplot as plt


# Read the csv file and extract labels for each image
labels = pd.read_csv('./Datasets/celeba/labels.csv', delimiter = "\t")

lab_names = labels["img_name"].values
lab_gen = labels["gender"].values



