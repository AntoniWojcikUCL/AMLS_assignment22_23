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
from PIL import Image

# Plotting libraries
import matplotlib.pyplot as plt



# Constants
DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "gender"

# Helper functions

def loadImgData(dataset_path, labels, img_per_batch, batch_iteration):
    img_data = []

    for i in range(img_per_batch):
        img = Image.open(dataset_path + '/img/' + labels[batch_iteration * img_per_batch + i])#.convert('L') # Open images and convert to greyscale
        img = np.array(img).flatten()
        
        img_data.append(img)

    img_data = np.array(img_data)

    return img_data

def unisonShuffleCopies(a, b, seed):
    assert len(a) == len(b)

    p = np.random.RandomState(seed = seed).permutation(len(a))
    return a[p], b[p]



### LEARNING

# Read the csv file and extract label_file for each image
label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")

labels = label_file[LABEL_IMG_NAMES].values
lab_gen = label_file[LABEL_NAME].values

# Read the images and preprocess them
img_count = len(labels)

#img_size = Image.open(DATASET_PATH + '/img/' + labels[0]).size

img_per_batch = 1000
num_batches = int(img_count / img_per_batch)

# Select the classifier 
clf = SGDClassifier(learning_rate = 'optimal', alpha = 1e-4, eta0 = 0.1, shuffle = False, loss = 'log_loss')

# Find the unique 
uq_classes = np.array(np.unique(lab_gen))

# Shuffle all the images
lab_gen, labels = unisonShuffleCopies(lab_gen, labels, seed = 42)

# Make the model learn on batches of images
for k in range(num_batches):

    print("Starting batch: ", k)

    img_data = loadImgData(DATASET_PATH, labels, img_per_batch, k)

    X_train = []
    y_train = []

    X_train = img_data
    y_train = lab_gen[k * img_per_batch:(k+1) * img_per_batch]

    # Learn the digits on the train subset
    clf.partial_fit(X_train, y_train, classes=uq_classes)


### TESTING

# Read the csv file and extract label_file for each image
label_file = pd.read_csv(TEST_DATASET_PATH + '/labels.csv', delimiter = "\t")

labels = label_file[LABEL_IMG_NAMES].values
lab_gen = label_file[LABEL_NAME].values

# Read the images and preprocess them
img_count = len(labels)

#img_size = Image.open(DATASET_PATH + '/img/' + labels[0]).size

img_per_batch = 1000
num_batches = int(img_count / img_per_batch)

y_predicted = np.zeros((img_count))

for k in range(num_batches):

    print("Starting test batch: ", k)

    img_data = loadImgData(TEST_DATASET_PATH, labels, img_per_batch, k)

    X_test = img_data

    # Learn the digits on the train subset
    predicted = clf.predict(X_test)

    # Predict the value of the digit on the test subset
    y_predicted[k * img_per_batch:(k+1) * img_per_batch] = predicted[:]


# Print the results
print("Labels: ", lab_gen)
print("Predicted: ", y_predicted)

score = np.sum(lab_gen == y_predicted) / img_count
print("Score: ", score)