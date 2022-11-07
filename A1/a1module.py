import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Image manipulation libraries
from PIL import Image

# Plotting libraries
import matplotlib.pyplot as plt


# Read the csv file and extract labels for each image
labels = pd.read_csv('./Datasets/celeba/labels.csv', delimiter = "\t")

lab_names = labels["img_name"].values
lab_gen = labels["gender"].values

# Read the images and preprocess them
img_count = len(lab_names)


img_size = Image.open("./Datasets/celeba/img/" + lab_names[0]).size

img_per_batch = 200
num_batches = int(img_count / img_per_batch)

# Apply GridSearchCV to find best parameters for given dataset
# verbose is used to describe the steps taken to find best parameters
#clf = SVC(gamma=0.001, kernel="rbf")
clf = SGDClassifier(learning_rate = 'constant', eta0 = 0.1, shuffle = False)

X_test = []
y_test = []

av_classes = np.array(np.unique(lab_gen))

for k in range(num_batches):

    print("Starting batch: ", k)

    img_data = []

    for i in range(img_per_batch):
        img_i = Image.open("./Datasets/celeba/img/" + lab_names[k * img_per_batch + i]).convert('L') # Open images and convert to greyscale
        
        img_data.append(np.array(img_i).flatten())

    X_train = []
    y_train = []

    img_data = np.array(img_data)

    if k < num_batches - 1:
        X_train = img_data
        y_train = lab_gen[k * img_per_batch:(k+1) * img_per_batch]
    else:
        # Split data into 70% train and 30% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            img_data, lab_gen[k * img_per_batch:(k+1) * img_per_batch], test_size = 0.2, shuffle = True
        )

    # Learn the digits on the train subset
    clf.partial_fit(X_train, y_train, classes=av_classes)


# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print("Labels: ", y_test)
print("Predicted: ", predicted)
print(clf.score(X_test, y_test))

#classifier.fit(x_train, y_train)
#clf.partial_fit(X_minibatch, y_minibatch)
#y_pred = classifier.predict(x_test)