# Assignment in Applied Machine Learning Systems ELEC0134 (22/23)

This project contains Python 3 code that solves the assignments provided in **ELEC0134(22-23)-new.pdf** assignment description that can be found on UCL Moodle website for the Applied Machine Learning Systems course.

In short, the problem focuses on use of machine learning for classification tasks. Tasks A1 and A2 deal with binary classification of facial characteristics of pre-processed images of celebrities: A1 - gender, A2 - whether person is smiling or not. Tasks B1 and B2 address multi-class classification of images of cartoon faces to: A1 - classify face shapes, B2 - classify eye colour of the characters.

More details about the tasks can be found in the assignment description.

## Installation

The code was written in Python 3 programming language, and it uses following packages at the provided versions:

- matplotlib      3.6.2
- numpy           1.23.5
- opencv-python   4.6.0.66
- pandas          1.5.2
- Pillow          9.3.0
- scikit-learn    1.2.0

To install the packages, install the packages manually or execute the following command in the project folder:

`python -m pip install -r requirements.txt`

## Running Code and Project Structure

The code is divided into scripts for each of the tasks described in the assignment, each placed in the individual folder bearing the name of the respective task. In each of these scripts there is main code solving the given task, which is contained in `run_task(...)` function. The function is executed automatically with default parameters when the script is run from command line. The function parameters slightly change the behaviour of the solution, as will be described in the subsections below. Aside from these scripts, there is a [main.py](main.py) script in the root folder, which runs all the other scripts in order, at different configurations, and outputs their total execution time. The data sets used in the tasks should be placed in the [Datasets](Datasets) folder and are not provided in this repository due to their sizes. They can be downloaded by following the full assignment description.

### Task A1

The main function of [a1module.py](A1/a1module.py) takes the following parameters with their default values:

`run_task(run_cross_val = True, clf_optimal_idx = 0, use_grayscale = True, show_mean = False, gen_convergence_plot = False, plot_out_path = "")`

- `run_cross_val` (*boolean*). When set to `True`, cross validation is used to select an optimal model (defined internally in the function) by comparing their accuracy scores. When set to `False`, the model with index `clf_optimal_idx` is taken from the list of models.
- `clf_optimal_idx` (*non-negative integer*).
- `use_grayscale` (*boolean*). When set to `True`, the loaded images will be transformed to grayscale, before being used by the model. This is done to speed up calcualtions and lower the size of the model.
- `show_mean` (*boolean*). When set to `True`, means and standard deviations of the images will be calculated (mean pixel value over the images, at each location respectively) and shown, for each class described by the labels.
- `gen_convergence_plot` (*boolean*). When set to `True`, convergence plots of the optimal model will be generated and stored in the path given by `plot_out_path`. 
- `plot_out_path` (*string*).

### Task A2

The main function of [a2module.py](A2/a2module.py) takes the following parameters with their default values:

`run_task(use_grayscale = True, show_mean = False, gen_convergence_plot = False, plot_out_path = "")`

The parameters have the same function as described for **Task A1**.

### Task B1

The main function of [b1module.py](B1/b1module.py) takes the following parameters with their default values:

`run_task(enable_edge_detection = True, enable_resize = True, resize_scaling = 0.5, show_mean = False, gen_convergence_plot = False, plot_out_path = "")`

- `enable_edge_detection` (*boolean*). When set to `True`, the loaded images are preprocessed with an edge detection algorithm.
- `enable_resize` (*boolean*). When set to `True`, the loaded images are rescaled by a factor given by the `resize_scaling` parameter. This is done to speed up calculations.
- `resize_scaling` (*float*).

The other parameters have the same function as described for **Task A1**.

### Task B2

The main function of [b2module.py](B2/b2module.py) takes the following parameters with their default values:

`run_task(add_sunglasses_lab = False, rm_train_sun_dp = True, rm_test_sun_dp = True, gen_convergence_plot = False, plot_out_path = "")`

- `add_sunglasses_lab` (*boolean*). When set to `True`, a new label marking images with characters wearing sunglasses is added to the training and test data sets.
- `rm_train_sun_dp` (*boolean*). When set to `True`, the data points with images with characters wearing sunglasses are removed from the **training** data set.
- `rm_test_sun_dp` (*boolean*). When set to `True`, the data points with images with characters wearing sunglasses are removed from the **test** data set.

The other parameters have the same function as described for **Task A1**.

## Code Structure

All the scripts used to solve the tasks follow a similar code structure, with the following classes and functions:
- `Timer` class, used to time execution of certain sections of the code. 
- `load_data_source(dataset_path, file_names)` function, used to load and pre-process images from the datasets from folder given in `dataset_path` (*string*) and names given in `dataset_path` (*list of strings*). 
- `load_Xy_data(dataset_path, ...)` function which outputs numpy arrays of X (pre-processed image data) and y (labels) data from the data sets specified in `dataset_path` (*string*), that can be used to train the models.
- `plot_convergence(clf, X, y, plot_out_path = "")` function which outputs a convergence plot of the model given by `clf` (*scikit-learn model*), trained on `X` and `y` (*numpy arrays*) data and labels. The plots are stored in path given by `plot_out_path` (*string*), if specified.
- `run_task(...)` function - the main function in every script. Depending on the script, the following steps are done in the function:
    1. Define the models (classifiers) that will be compared, trained and used to test new data.
    2. Load training data using `load_Xy_data(...)`.
    3. *(Only done in tasks A1 and A2)* Run cross validation to select the best model from the list specified in *1*, based on their accuracy scores.
    4. Train the best model on all the training data.
    5. Use cross-validation to generate a convergence plot for the best model using `plot_convergence(...)`. 
    6. Load test data using `load_Xy_data(...)`.
    7. Obtain predictions from the trained model. Then print the accuracy scores and plot the confusion matrices.

More details about each script can be found in the comments provided in the code.

