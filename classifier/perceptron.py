from .classifier import *
import random
import numpy as np


""" 
00 01 02 03
10 11 12 13
We know the following:
digit width = 28
digit height = 28
face width = 60
face height = 70
"""

class Perceptron(Classifier):
    def __init__(self, legal_labels):
        super().__init__(legal_labels)
        self.max_iterations = 10
        self.weights = {}
        self.biases = {}
        digit = True if len(legal_labels) == 10 else False

        if digit:
            for label in legal_labels:
                self.weights[label] = np.array([random.uniform(0, 1) for i in range(28 * 28)])
                self.biases[label] = random.uniform(0, 1)
        else:  # face
            for label in legal_labels:
                self.weights[label] = np.array([random.uniform(0, 1) for i in range(70 * 60)])
                self.biases[label] = random.uniform(0, 1)
                        
    @staticmethod
    def name():
        return "perceptron"
        
    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def _separate(self, dataset):
        classes = {}  # arrays of images belonging to a particular class
        inverse_classes = {}  # arrays of images specifically NOT belonging to a class
        for label in self.legalLabels:
            classes[label] = []
            inverse_classes[label] = []

        for data, cls in dataset:
            # add this image to the array for its class
            classes[cls].append(np.array(data.flat_data()))
            for label in self.legalLabels:
                # add it to the "inverse" set of images for all other labels
                if label != cls:
                    inverse_classes[label].append(np.array(data.flat_data()))
        return classes, inverse_classes

    def _classify_img(self, flat_img, bias, weights):
        total = bias + (flat_img * weights).sum()
        return 1 / (1 + np.math.exp(-0.1 * total))

    def train(self, training_data, validation_data):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """
        groups, inverses = self._separate(training_data.get_labeled_images())

        alpha = 0.05  # training rate
        iterations = 14  # number of iterations
        for i in range(iterations):
            for cls in self.legalLabels:
                # iterate through each image belonging to this class
                for image in groups[cls]:
                    # compute the error 1 (meaning 'true' this image is in this class) - calculated value
                    error = 1 - self._classify_img(image, self.biases[cls], self.weights[cls])
                    # adjust bias (or activation)
                    self.biases[cls] += alpha * error
                    # adjust weights factoring in training rate
                    self.weights[cls] = self.weights[cls] + alpha * error * image

                for image in inverses[cls]:
                    # compute the error (0 means not belonging to this class)
                    error = 0 - self._classify_img(image, self.biases[cls], self.weights[cls])
                    # adjust bias (or activation)
                    self.biases[cls] += alpha * error
                    # adjust weights factoring in training rate
                    self.weights[cls] += alpha * error * image

        return self.validate(validation_data)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for image, label in data.get_labeled_images():
            flat_img = image.flat_data()
            class_activations = []
            for cls in self.weights.keys():
                # calculate the percent chance this image belongs to this class
                activation = self._classify_img(flat_img, self.biases[cls], self.weights[cls])
                # add that chance to the list of % chance for each class
                class_activations.append((cls, activation))
            # pick the class with the highest value, and add it to the list of guesses
            guesses.append(max(class_activations, key=lambda x: x[1])[0])
        return guesses