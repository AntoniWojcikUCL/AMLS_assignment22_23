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


DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'


### LEARNING

# Read the csv file and extract labels for each image
labels = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")

lab_names = labels["img_name"].values
lab_gen = labels["gender"].values

# Read the images and preprocess them
img_count = len(lab_names)

#img_size = Image.open(DATASET_PATH + '/img/' + lab_names[0]).size

img_per_batch = 1000
num_batches = int(img_count / img_per_batch)

# Apply GridSearchCV to find best parameters for given dataset
# verbose is used to describe the steps taken to find best parameters
#clf = SVC(gamma=0.001, kernel="rbf")
clf = SGDClassifier(learning_rate = 'optimal', eta0 = 0.1, shuffle = True)

av_classes = np.array(np.unique(lab_gen))


# Shuffle all the images

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

lab_gen, lab_names = unison_shuffled_copies(lab_gen, lab_names)


for k in range(num_batches):

    print("Starting batch: ", k)

    img_data = []

    for i in range(img_per_batch):
        img = Image.open(DATASET_PATH + '/img/' + lab_names[k * img_per_batch + i]) # Open images and convert to greyscale
        img = np.array(img).flatten()
        
        img_data.append(img)

    X_train = []
    y_train = []

    img_data = np.array(img_data)

    X_train = img_data
    y_train = lab_gen[k * img_per_batch:(k+1) * img_per_batch]

    # Learn the digits on the train subset
    clf.partial_fit(X_train, y_train, classes=av_classes)


### TESTING

# Read the csv file and extract labels for each image
labels = pd.read_csv(TEST_DATASET_PATH + '/labels.csv', delimiter = "\t")

lab_names = labels["img_name"].values
lab_gen = labels["gender"].values

# Read the images and preprocess them
img_count = len(lab_names)

#img_size = Image.open(DATASET_PATH + '/img/' + lab_names[0]).size

img_per_batch = 1000
num_batches = int(img_count / img_per_batch)

y_predicted = np.zeros((img_count))

for k in range(num_batches):

    print("Starting test batch: ", k)

    img_data = []

    for i in range(img_per_batch):
        img = Image.open(TEST_DATASET_PATH + '/img/' + lab_names[k * img_per_batch + i]) # Open images and convert to greyscale
        img = np.array(img).flatten()
        
        img_data.append(img)

    img_data = np.array(img_data)

    X_test = img_data

    # Learn the digits on the train subset
    predicted = clf.predict(X_test)

    # Predict the value of the digit on the test subset
    y_predicted[k * img_per_batch:(k+1) * img_per_batch] = predicted[:]


print("Labels: ", lab_gen)
print("Predicted: ", y_predicted)

score = np.sum(lab_gen == y_predicted) / img_count
print("Score: ", score)

#classifier.fit(x_train, y_train)
#clf.partial_fit(X_minibatch, y_minibatch)
#y_pred = classifier.predict(x_test)