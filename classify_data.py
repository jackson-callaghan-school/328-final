import glob
import sys
import numpy as np
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
import sklearn

class_names = ["falling", "jumping", "sitting", "standing", "turning", "walking"]

print("Collecting data from /data directory...")
sys.stdout.flush()

file_names = glob.glob("data/*")

data = np.genfromtxt(file_names[0], delimiter=',')

for i in range(1, len(file_names)):
    data = np.concatenate((data, np.genfromtxt(file_names[i], delimiter=',')),axis=0)

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i, 1], data[i, 2], data[i, 3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1], reoriented, axis=1)

print("Rebuilding data with gyroscope and class...")
sys.stdout.flush()
# Handles adding back gyroscope data and class
data = np.append(reoriented_data_with_timestamps, data[:,-4:], axis=1)

window_size = 20

step_size = 20

n_samples = 100

time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

sampling_rate = n_samples / time_elapsed_seconds

print("The sampling rate is {} Hz".format(sampling_rate))
sys.stdout.flush()

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i, window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:, 1:-1]
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])

X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# Train the data

cv = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

accuracy = np.zeros(10)
counter = 0

for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
    tree.fit(X_train, Y_train)
    Y_pred = tree.predict(X_test)
    conf = sklearn.metrics.confusion_matrix(Y_test, Y_pred)

    print(conf)
    print('Accuracy Score :', sklearn.metrics.accuracy_score(Y_test, Y_pred))
    accuracy[counter] = sklearn.metrics.accuracy_score(Y_test, Y_pred)
    counter+=1
    print('Report : ')
    print(sklearn.metrics.classification_report(Y_test, Y_pred))

print("average accuracy = " + str(np.mean(accuracy)))

activity_classifier = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
activity_classifier = activity_classifier.fit(X, Y)

# Some additional info about the trained model

print("Stats for how well the model does on its own data:")

Y_full_pred = activity_classifier.predict(X)
conf = sklearn.metrics.confusion_matrix(Y, Y_full_pred)

print(conf)
print('Accuracy Score :', sklearn.metrics.accuracy_score(Y, Y_full_pred))
print('Report : ')
print(sklearn.metrics.classification_report(Y, Y_full_pred))

print("Saving classification as 'tree.dot', 'classifier.pickle'")

export_graphviz(activity_classifier, out_file='tree.dot', feature_names=feature_names)

with open('classifier.pickle', 'wb') as f:
    pickle.dump(activity_classifier, f)
