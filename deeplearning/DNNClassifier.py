import tensorflow as tf
import numpy as np
from six.moves.urllib.request import urlretrieve 

# Data sets
# Download just once
# trainingset_url = "http://download.tensorflow.org/data/iris_training.csv"
# testset_url = "http://download.tensorflow.org/data/iris_test.csv"
trainingset_filename = "iris_training.csv"
testset_filename = "iris_test.csv"
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
"""
print("Attempting to download IRIS datasets ...\n")
IRIS_TRAINING, _ = urlretrieve(trainingset_url, trainingset_filename)
IRIS_TEST, _ = urlretrieve(testset_url, testset_filename)
print("Download Completed!!\n")
"""

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target

print(x_train)

# print("Training set shape: {}".format(str(training_set.data.shape)))
# print("Test set shape: {}".format(str(test_set.data.shape)))

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

# Fit model.
classifier.fit(x=x_train, y=y_train, steps=200)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
# print(accuracy_score)

print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[7.4, 3.2, 5.4, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))